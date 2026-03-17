# python3 -c "
# import ast
# src = open('examples/on_policy_distillation/on_policy_self_distillation.py', encoding='utf-8').read()
# ast.parse(src)
# print('Syntax OK')

# Check what mask_token Qwen3 tokenizer exposes (if available on this machine)
try:
    from transformers import AutoTokenizer
    # Try Qwen3-8B first, then Qwen3-4B
    import os
    for path in [ 'Qwen/Qwen2.5-7B-Instruct']:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        print(f'{path}: mask_token={repr(tok.mask_token)}, eos={repr(tok.eos_token)}')
        break
    else:
        print('No local model found; mask_token check skipped.')
except Exception as e:
    print(f'Tokenizer check skipped: {e}')
