"""Stable generation profile with explicit decoding, role history and raw evidence.
No canned correct contract, no template policies, no post-generation action edits.
"""
import time
from core.cover_cabto_grounding import LocalModel


class StableModel(LocalModel):
    def call(self,text,images=(),max_tokens=2400,temperature=0.0):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        if isinstance(text,str):
            messages=[{"role":"system","content":"You are a careful robotics programmer. Follow the requested output format exactly. Produce short complete answers; never repeat code or copy feedback verbatim."},{"role":"user","content":text}]
        else: messages=text
        prompt=apply_chat_template(self.processor,self.config,messages,num_images=len(images),enable_thinking=False)
        settings={"max_tokens":max_tokens,"temperature":max(.15,temperature),"top_p":.9,"top_k":20,
                  "repetition_penalty":1.08,"repetition_context_size":256,"seed":1701}
        start=time.perf_counter()
        out=generate(self.model,self.processor,prompt,image=[str(p) for p in images] if images else None,verbose=False,**settings)
        return {"model_id":self.model_id,"messages":messages,"image_files":[str(p) for p in images],
                "formatted_prompt":prompt,"decoding":settings,"raw":out.text if hasattr(out,"text") else str(out),
                "generation_tokens":getattr(out,"generation_tokens",None),"wall_seconds":time.perf_counter()-start}


class ThinkingModel(LocalModel):
    def call(self,text,images=(),max_tokens=5000,temperature=.6):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        prompt=apply_chat_template(self.processor,self.config,text,num_images=len(images),enable_thinking=True)
        settings={"max_tokens":max_tokens,"temperature":max(.6,temperature),"top_p":.95,"top_k":20,
                  "repetition_penalty":1.05,"repetition_context_size":512,"seed":1701}
        start=time.perf_counter()
        output=generate(self.model,self.processor,prompt,image=[str(p) for p in images] if images else None,verbose=False,**settings)
        return {"model_id":self.model_id,"messages":text,"formatted_prompt":prompt,"image_files":[str(p) for p in images],
                "enable_thinking":True,"decoding":settings,"raw":output.text,"generation_tokens":output.generation_tokens,
                "finish_reason":output.finish_reason,"wall_seconds":time.perf_counter()-start}
