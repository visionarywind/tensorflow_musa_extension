import os
import re

# 配置路径
TARGET_DIR = "/Users/shanfeng/Workspace/Moore/tensorflow_musa_extension/musa_ext/kernels"

# 要插入的代码片段 (注意前面的换行和缩进)
# 我们将日志块定义为多行字符串，确保缩进一致
LOG_CODE_STR = """
  static bool debug_log = std::getenv("MUSA_KERNEL_DEBUG_LOG") == nullptr;
  if (debug_log) {
    std::stringstream ss;
    ss << "[MUSA Debug] Thread: " << std::this_thread::get_id()
              << " | Op: " << __FILE__
              << " | Method: " << __FUNCTION__;
    int input_num = ctx->num_inputs();
    for (int i = 0; i < input_num; ++i) {
        ss << " | Input " << i << ": " << ctx->input(i).shape().DebugString();
    }
    LOG(ERROR) << ss.str();
  }
  static bool sync_execute = std::getenv("MUSA_LAUNCH_BLOCKING") == "1";
  if (sync_execute) {
    musaStreamSynchronize(GetMusaStreamByCtx(ctx));
  }
"""

# 需要确保包含的头文件
REQUIRED_HEADERS = [
    "#include <thread>",
    "#include <tensorflow/core/platform/logging.h>",
    "#include <cstdlib>"
]

def add_headers_if_missing(content):
    """检查并添加缺失的头文件"""
    lines = content.split('\n')
    existing_includes = {line.strip() for line in lines if line.strip().startswith("#include")}
    
    headers_to_add = [h for h in REQUIRED_HEADERS if h not in existing_includes]
    
    if not headers_to_add:
        return content, False

    # 找到最后一个 include 的位置
    last_include_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("#include"):
            last_include_idx = i
            
    if last_include_idx == -1:
        new_lines = headers_to_add + [''] + lines
        return "\n".join(new_lines), True

    new_lines = lines[:last_include_idx+1] + headers_to_add + lines[last_include_idx+1:]
    return "\n".join(new_lines), True

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # 1. 幂等性检查
        if "MUSA_KERNEL_DEBUG_LOG" in original_content:
            return False

        content = original_content
        modified = False

        # 2. 处理头文件
        new_content, header_modified = add_headers_if_missing(content)
        if header_modified:
            content = new_content
            modified = True

        # 3. 核心逻辑：匹配 Compute 方法
        # 策略：
        # 1. 匹配整个方法签名直到左大括号 {
        # 2. 将 { 替换为 {\n<LOG_CODE>
        # 这样能保证日志代码绝对在方法体的第一行
        
        # 正则解释：
        # (void\s+[\s\S]*?::Compute\s*\([\s\S]*?\)\s*(?:const\s*)?(?:override\s*)?\s*)
        # 上述分组捕获了从 void 到 { 之前的所有内容（包括可能的换行和空格）
        # \{ 匹配左大括号
        
        pattern = r'(void\s+[\s\S]*?Compute\s*\([\s\S]*?\)\s*?(?:override\s*)?\s*)\{'
        
        match_count = 0
        def replace_compute(match):
            nonlocal modified, match_count
            modified = True
            match_count += 1
            
            # group(1) 是方法签名部分
            # 我们保留 group(1)，然后加上 {，再换行，再加日志代码
            # 注意：LOG_CODE_STR 开头已经有一个换行符，所以这里直接拼接即可
            return f"{match.group(1)}{{\n{LOG_CODE_STR}"

        new_content = re.sub(pattern, replace_compute, content, flags=re.DOTALL)

        if modified and match_count > 0:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"[OK] Patched ({match_count} methods): {os.path.basename(filepath)}")
            return True
        else:
            return False

    except Exception as e:
        print(f"[ERR] Failed {filepath}: {e}")
        return False

def main():
    if not os.path.exists(TARGET_DIR):
        print(f"Directory not found: {TARGET_DIR}")
        return

    processed_count = 0
    patched_count = 0
    
    for root, dirs, files in os.walk(TARGET_DIR):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'build']
        
        for file in files:
            if file.endswith('.cc'):
                filepath = os.path.join(root, file)
                if process_file(filepath):
                    patched_count += 1
                processed_count += 1
    
    print(f"\nSummary:")
    print(f"Scanned: {processed_count} .cc files")
    print(f"Patched: {patched_count} files")
    print("\nTo enable logs, run your program with:")
    print("export MUSA_KERNEL_DEBUG_LOG=1")

if __name__ == "__main__":
    main()