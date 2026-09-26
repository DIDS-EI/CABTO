"""Qwen semantic keypoint + top-down RGB geometric refinement.
No segmentation renderer, geom IDs, object positions, GT choice or oracle fallback.
Uses explicit color/category priors and a calibrated known height plane. This is
hybrid VLM+classical vision, NOT unaided VLM pointing or pure sensor-only control.
"""
from pathlib import Path
import json
import cv2
import numpy as np
import mujoco
from PIL import Image, ImageDraw
from core.cover_cabto_api import Perception, intersect_plane
from core.cover_cabto_grounding import save

CAMERA = {"name": "front", "position": [.47,0.,1.5], "quat": [1.,0.,0.,0.], "fovy":45.}
DESCRIPTIONS = {"shrimp":"the orange shrimp", "apple":"the red apple", "potato":"the small yellow cuboid with an oval top, not the orange shrimp",
                "bowl":"the center of the white bowl interior", "board":"the center of the wooden chopping board",
                "pan":"the center of the round black pan interior, excluding its handle"}


def install_camera(env):
    i=mujoco.mj_name2id(env.m,mujoco.mjtObj.mjOBJ_CAMERA,CAMERA["name"])
    before={"position":env.m.cam_pos[i].tolist(),"quat":env.m.cam_quat[i].tolist(),"fovy":float(env.m.cam_fovy[i])}
    env.m.cam_pos[i]=CAMERA["position"];env.m.cam_quat[i]=CAMERA["quat"];env.m.cam_fovy[i]=CAMERA["fovy"]
    mujoco.mj_forward(env.m,env.d)
    return {"before":before,"after":CAMERA,"change":"observation-only camera extrinsics; no physical scene/control change"}


def refine_rgb(image, coarse_uv, name):
    """Return refined pixel from RGB and model point alone. Fail closed on ambiguity."""
    h,w=image.shape[:2];uv=np.asarray(coarse_uv,float)
    if uv.shape!=(2,) or not np.isfinite(uv).all() or np.any(uv<0) or uv[0]>=w or uv[1]>=h:
        raise ValueError("Model point outside frame")
    radius=84 if name in ("bowl","board","pan") else 36
    x0,y0=max(0,int(uv[0])-radius),max(0,int(uv[1])-radius)
    x1,y1=min(w,int(uv[0])+radius+1),min(h,int(uv[1])+radius+1)
    crop=image[y0:y1,x0:x1];hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
    hue,sat,val=[hsv[:,:,i].astype(float) for i in range(3)]
    if name=="bowl":mask=(sat<42)&(val>190)
    elif name=="pan":mask=(sat<65)&(val<67)
    elif name=="apple":mask=((hue<10)|(hue>170))&(sat>95)&(val>45)
    elif name in ("board","potato"):mask=(hue>12)&(hue<42)&(sat>40)&(val>90)
    elif name=="shrimp":mask=(hue<42)&(sat>65)&(val>110)
    else:raise ValueError("Unknown semantic color prior")
    mask=mask.astype(np.uint8)
    if name=="pan":
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)))
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
    candidates=[]
    for i in range(1,n):
        area=int(stats[i,cv2.CC_STAT_AREA]);center=centroids[i]+[x0,y0]
        if area<(180 if name in ("bowl","board","pan") else 45):continue
        distance=float(np.linalg.norm(center-uv))
        if distance<radius*.85:candidates.append((distance,i,area))
    if not candidates:raise ValueError("No RGB component agrees with Qwen semantic point")
    candidates.sort()
    if len(candidates)>1 and candidates[1][0]-candidates[0][0]<6:
        raise ValueError("Ambiguous RGB components near Qwen point; re-observe instead of guessing")
    _,label,area=candidates[0]
    component=(labels==label).astype(np.uint8)
    if name=="apple":
        from scipy.ndimage import binary_fill_holes
        component=binary_fill_holes(component).astype(np.uint8)
    if name in ("bowl","board"):
        # The complete support footprint is more stable than brightest-pixel maxima.
        contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour=max(contours,key=cv2.contourArea)
        if name=="bowl" and len(contour)>=5:
            ellipse=cv2.fitEllipse(contour);local=np.array(ellipse[0]);method="white_component_outer_ellipse_center"
        else:
            moments=cv2.moments(component);local=np.array([moments["m10"]/moments["m00"],moments["m01"]/moments["m00"]]);method="component_centroid"
    else:
        # Maximum inscribed disc rejects the thin pan handle and shrimp tail.
        distance=cv2.distanceTransform(component,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        ys,xs=np.where(distance>=distance.max()-.25)
        local=np.array([xs.mean(),ys.mean()]);method="maximum_inscribed_disc"
    if name=="pan":
        grey=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
        circles=cv2.HoughCircles(cv2.GaussianBlur(grey,(5,5),1),cv2.HOUGH_GRADIENT,1,30,
                                 param1=80,param2=28,minRadius=25,maxRadius=78)
        if circles is not None:
            candidates=[c for c in circles[0] if np.linalg.norm(c[:2]+[x0,y0]-uv)<radius*.8]
            if not candidates:raise ValueError("Round pan edge does not match semantic point")
            circle=min(candidates,key=lambda c:np.linalg.norm(c[:2]+[x0,y0]-uv))
            local=circle[:2].astype(float);radius_est=float(circle[2])
            ey,ex=np.where(cv2.Canny(grey,40,80)>0)
            points=np.column_stack((ex,ey)).astype(float)
            for _ in range(3):
                distances=np.linalg.norm(points-local,axis=1)
                inliers=points[np.abs(distances-radius_est)<2.5]
                if len(inliers)<24:break
                design=np.column_stack((2*inliers[:,0],2*inliers[:,1],np.ones(len(inliers))))
                solution=np.linalg.lstsq(design,np.sum(inliers**2,axis=1),rcond=None)[0]
                local=solution[:2];radius_est=float(np.sqrt(max(0,solution[2]+local@local)))
            method="round_edge_hough_center"
    refined=local+[x0,y0]
    if np.linalg.norm(refined-uv)>radius*.8:raise ValueError("Geometric refinement too far from Qwen prediction")
    # Reject clipped components; do not infer an unseen object's center from a partial blob.
    if component[0].any() or component[-1].any() or component[:,0].any() or component[:,-1].any():
        raise ValueError("RGB component crosses ROI border; no reliable complete shape")
    return refined,{"method":method,"roi":[x0,y0,x1,y1],"component_area":area,
                    "coarse_uv":uv.tolist(),"refined_uv":refined.tolist(),"semantic_prior":name},component


class OverheadPerception:
    def __init__(self, runtime, out, client):
        self.r=runtime;self.out=Path(out);self.client=client;self.records=[];self.cache={}
        self.camera_change=install_camera(runtime.env)
        self.out.mkdir(parents=True,exist_ok=True);save(self.out/"camera.json",self.camera_change)

    def locate(self,name,z,kind):
        if name in self.cache:
            p=self.cache[name].copy();self.records.append({"object":name,"cached":True,"xyz":p.tolist(),"origin":"initial RGB/Qwen static support","fallback":False});return p
        image=self.r.env.render_cam("front").copy();size=image.shape[0]
        index=len(self.records);path=self.out/f"{index:03d}_{name}.png";Image.fromarray(image).save(path)
        absolute="Qwen2.5-VL" in self.client.model_id
        text=(f"Point to {DESCRIPTIONS[name]}. Output the pixel coordinate." if absolute else
              f"Point to {DESCRIPTIONS[name]}. Output only JSON {{\"point\":[x,y]}}. Use normalized coordinates 0 to 1000, origin at image top-left.")
        row={"object":name,"point_kind":kind,"height_prior":float(z),"model_id":self.client.model_id,
             "source":"Qwen semantic point + RGB refinement + calibrated plane","fallback":False,"image":str(path)}
        self.records.append(row)
        try:
            call=self.client.call(text,[path],max_tokens=200);row["call"]=call
            xy,field=Perception.parse_pixel(call,size if absolute else 1000)
            coarse=xy if absolute else xy*size/1000;row["field"]=field
            row["coordinate_contract"]="absolute_pixel" if absolute else "normalized_0_1000"
            uv,diagnosis,mask=refine_rgb(image,coarse,name);row["refinement"]=diagnosis
            # The detected bowl/pan silhouette is at the RIM, not the floor.
            # Explicit vertical geometry priors from the frozen scene design;
            # never read the support x/y or a geom ID to correct the estimate.
            feature_z = self.r.config["table_top"] + {"bowl":.054,"pan":.039}[name] if name in ("bowl","pan") else z
            p=intersect_plane(self.r.env,uv,"front",size,feature_z)
            row["observed_feature_plane_z"]=float(feature_z)
            row["feature_height_source"]="frozen manual rim geometry: bowl .029+.025, pan .021+.018; no xy prior"
            p[2]=z
            if not (.05<p[0]<.9 and -.36<p[1]<.36):raise ValueError("Target outside unchanged safe workspace")
            row["xyz"]=p.tolist();row["uv"]=uv.tolist()
            overlay=Image.fromarray(image);d=ImageDraw.Draw(overlay)
            for pt,color in ((coarse,(245,194,79)),(uv,(85,232,187))):
                u,v=pt;d.line((u-7,v,u+7,v),fill=color,width=2);d.line((u,v-7,u,v+7),fill=color,width=2)
            overlay.save(path.with_name(path.stem+"_overlay.png"));Image.fromarray(mask*255).save(path.with_name(path.stem+"_mask.png"))
            if kind=="support_floor_center":self.cache[name]=p.copy()
            return p
        except Exception as exc:row["error"]=str(exc);raise
        finally:save(self.out/"perception.json",self.records)

    def prepare_supports(self,names):
        for name in names:
            z=self.r.config["table_top"]+self.r.config["support_surface_z"][name]
            self.locate(name,z,"support_floor_center")
