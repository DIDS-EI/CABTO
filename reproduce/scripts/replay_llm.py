"""
ReplayLLM — 离线重放 LLM 客户端
================================
替代 btgym.llm.llm_gpt.LLM。它不联网、不调用任何 API，
而是从论文作者已经保存好的 a_exp1_llm_bt_results/<task>_try_<i>_<ts>/llm_output.txt
读取当时真实 LLM (gpt-4o / gpt-3.5) 的输出并原样返回。

这样可以在【零 API key、零网络、零费用】的前提下，
完整驱动原始实验脚本的"代码抽取 -> 行为库写盘 -> obtea 规划 -> sound&complete 验证"流程，
从而复现论文实验1 (High-level model proposal) 的 first-proposal 成功率结果。

接口与原 LLM 保持一致:
    llm = ReplayLLM(request_model="gpt-4o")
    llm.set_context(task_name, try_idx)   # 复现脚本在每个 try 前调用
    answer = llm.request(prompt_or_messages)
"""
import os
import re
import glob

# llm_output.txt 的前缀标记，需要剥离
_OUTPUT_HEADER = "=== LLM Output ==="


class ReplayLLM:
    def __init__(self, results_dir, request_model="gpt-4o", timestamp=None, verbose=False):
        """
        results_dir : 存放 <task>_try_<i>_<ts>/ 的目录 (a_exp1_llm_bt_results)
        request_model: 仅用于日志，重放不真正使用
        timestamp   : 指定批次时间戳 (如 '202511171622')；None 时自动探测
        """
        self.results_dir = results_dir
        self.request_model = request_model
        self.verbose = verbose
        self._cur_task = None
        self._cur_try = None
        self._timestamp = timestamp or self._detect_timestamp()
        if self.verbose:
            print(f"[ReplayLLM] results_dir={results_dir} timestamp={self._timestamp}")

    # ---- 自动探测结果目录里使用的时间戳后缀 ----
    def _detect_timestamp(self):
        dirs = glob.glob(os.path.join(self.results_dir, "task*_try_*_*"))
        for d in dirs:
            m = re.search(r"task\d+_try_\d+_(\d+)$", os.path.basename(d))
            if m:
                return m.group(1)
        return None

    def set_context(self, task_name, try_idx):
        """复现脚本在每个 (task, try) 之前调用，告诉重放器读哪个目录。"""
        self._cur_task = task_name
        self._cur_try = try_idx

    def _try_dir(self):
        # 命名形如 task3_try_1_202511171622
        cand = os.path.join(
            self.results_dir,
            f"{self._cur_task}_try_{self._cur_try}_{self._timestamp}",
        )
        if os.path.isdir(cand):
            return cand
        # 回退：模糊匹配（时间戳可能因 task 而异）
        pat = os.path.join(self.results_dir, f"{self._cur_task}_try_{self._cur_try}_*")
        hits = sorted(glob.glob(pat))
        return hits[0] if hits else None

    def _read_cached_output(self):
        d = self._try_dir()
        if d is None:
            raise FileNotFoundError(
                f"[ReplayLLM] 找不到缓存目录: {self._cur_task}_try_{self._cur_try}_* in {self.results_dir}"
            )
        out_path = os.path.join(d, "llm_output.txt")
        if not os.path.isfile(out_path):
            raise FileNotFoundError(f"[ReplayLLM] 缺少 llm_output.txt: {out_path}")
        with open(out_path, "r", encoding="utf-8") as f:
            text = f.read()
        # 剥离保存时加的 header
        idx = text.find(_OUTPUT_HEADER)
        if idx != -1:
            text = text[idx + len(_OUTPUT_HEADER):]
        return text.lstrip("\n")

    # ---- 与原 LLM.request 同名同签名 ----
    def request(self, message):
        """message 可以是 str(首轮) 或 messages list(反馈轮)。
        重放只保存了最后一次输出，因此两种情况都返回同一份缓存输出。"""
        answer = self._read_cached_output()
        if self.verbose:
            preview = answer[:80].replace("\n", " ")
            print(f"[ReplayLLM] replay {self._cur_task} try{self._cur_try}: {preview}...")
        return answer

    # 兼容可能存在的流式接口
    def stream_request(self, messages):
        return self.request(messages)
