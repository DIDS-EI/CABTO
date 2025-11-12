import grpc
import sys
from btgym import cfg
sys.path.append(f'{cfg.ROOT_PATH}/molmo')
import btgym.molmo.molmo_pb2 as molmo_pb2
import btgym.molmo.molmo_pb2_grpc as molmo_pb2_grpc
import numpy as np
import os
import re
from PIL import Image, ImageDraw
class MolmoClient:
    def __init__(self):
        i = 1
        while True: 
            try:
                self.channel = grpc.insecure_channel('localhost:50053')
                # 设置5秒超时
                grpc.channel_ready_future(self.channel).result(timeout=5)
                self.stub = molmo_pb2_grpc.MolmoServiceStub(self.channel)
                print(f"连接Molmo成功！")
                break
            except grpc.FutureTimeoutError:
                print(f"连接Molmo超时，重试第{i}次")
            except Exception as e:
                print(f"连接Molmo错误，错误信息: {str(e)}，重试第{i}次")
            i += 1

    def call(self, func,**kwargs):
        try:
            if kwargs:
                request = getattr(molmo_pb2, func+"Request")(**kwargs)
            else:
                request = molmo_pb2.Empty()
            response = getattr(self.stub, func)(request)
            return response
        except Exception as e:
            print(f"调用Molmo函数失败，错误信息: {str(e)}")
            return None

    def extract_points(self, molmo_output, image):
        all_points = []
        for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', molmo_output):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                point = point * np.array([image.width, image.height])
                all_points.append(point)
        return all_points

    def draw_points_on_image(self, image, points, output_path):
        # 创建图片副本以免修改原图
        img_with_points = image.copy()
        
        # 转换为可绘制格式
        draw = ImageDraw.Draw(img_with_points)
        
        radius = 10
        # 为每个点画一个红色圆圈和序号
        for i, point in enumerate(points):
            x, y = point
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill='green')
            draw.text((x-3, y-6), str(i+1), fill='white', font=None)
        # 保存并显示结果
        img_with_points.convert('RGB').save(output_path)
        # img_with_points.save(output_path)

    def get_grasp_pose_by_molmo(self,query,dir,point_img_path=None):
        # query = f'point out the grasp point of the {cfg.target_object_name.split(".")[0]}. make sure the grasp point is in a stable position and safe.'

        print('query:',query)
        if point_img_path==None:
            image_path = f'{dir}/camera_0_rgb.png'
        else:
            image_path = point_img_path

        generated_text = self.call(func='PointQA',
                        #    query=f'Point out the important parts for doing the task. The task is "reorient the white pen and drop it upright into the black pen holder".',
                        query=query,
                        image_path=image_path
                        ).text

        image = Image.open(image_path)
        points = self.extract_points(generated_text, image)
        print('molmo points',points)
        
        if point_img_path==None:
            self.draw_points_on_image(image, points, f'{dir}/camera_0_rgb_points.png')
        else:
            self.draw_points_on_image(image, points, point_img_path)

        if len(points) > 0:
            return int(points[0][0]),int(points[0][1])
        else:
            print('molmo没有标出任何点！')
            return None

        # response = client.call(func='SetCameraLookatPos', pos=pos)



def main():
    client = MolmoClient()
    DIR = os.path.dirname(os.path.abspath(__file__))
    # 测试加载任务
    # response = client.call(func='LoadTask', task_name='putting_shoes_on_rack')
    # response = client.call(func='GetTaskObjects')
    # print(response)
    response = client.call(func='PointQA',
                           query='reorient the white pen and drop it upright into the black pen holder.',
                           image_path=f'{DIR}/camera_0_rgb.png',
                        )
    print(response)

if __name__ == '__main__':
    main() 