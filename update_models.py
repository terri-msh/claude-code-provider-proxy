import urllib.request
import json
import re

def parse_version(model_id):
    model_id = model_id.split('/')[-1]
    version_str = re.sub(r'(\d)[-.](\d)', r'\1.\2', model_id)
    match = re.search(r'(\d+(?:\.\d+)?)', version_str)
    return float(match.group(1)) if match else 0.0

def get_family_and_tier(m):
    model_id = m.get('model_version', '').lower().split('/')[-1]
    
    if 'gpt' in model_id and not model_id.startswith('o') and 'oss' not in model_id:
        tier = 'pro' if 'pro' in model_id else 'base'
        return 'gpt', tier, parse_version(model_id)
        
    if 'gemini' in model_id:
        tier = 'flash' if 'flash' in model_id else 'pro' if 'pro' in model_id else 'base'
        return 'gemini', tier, parse_version(model_id)
        
    if 'claude' in model_id:
        tier = 'haiku' if 'haiku' in model_id else 'sonnet' if 'sonnet' in model_id else 'opus' if 'opus' in model_id else 'base'
        return 'claude', tier, parse_version(model_id)
        
    if 'grok' in model_id:
        tier = 'base'
        return 'grok', tier, parse_version(model_id)
        
    if re.match(r'^o\d+', model_id) or 'o4' in model_id:
        tier = 'pro' if 'pro' in model_id else 'base'
        return 'o-series', tier, parse_version(model_id)
        
    if 'qwen' in model_id:
        tier = 'coder' if 'coder' in model_id else 'max' if 'max' in model_id else 'turbo' if 'turbo' in model_id else 'plus' if 'plus' in model_id else 'base'
        return 'qwen', tier, parse_version(model_id)

    if 'deepseek' in model_id:
        tier = 'base'
        return 'deepseek', tier, parse_version(model_id)

    if 'glm' in model_id:
        tier = 'base'
        return 'glm', tier, parse_version(model_id)
        
    if 'kimi' in model_id:
        tier = 'dev' if 'dev' in model_id else 'base'
        return 'kimi', tier, parse_version(model_id)

    return 'other', 'base', parse_version(model_id)

def is_unwanted(m, is_internal=False):
    v = m.get('model_version', '').lower()
    f = m.get('model_family', '').lower()
    vendor = m.get('vendor', '').lower()
    handle = m.get('handle', '').lower()
    
    # Вендоры (для internal_models мы не фильтруем вендора 'internal')
    unwanted_vendors = [
        'ideogram', 'elevenlabs', 'imaginepro', 'genapi', 
        'recraft', 'serpapi', 'searchapi', 'search',
        'perplexity', 'cohere'
    ]
    if not is_internal:
        unwanted_vendors.extend(['gigachat', 'internal'])
        
    if vendor in unwanted_vendors:
        return True
        
    # Исключаем конкретные слова (test, embeddings и т.д. должны отсеиваться и из internal)
    unwanted_keywords = [
        'llama', 'mixtral', 'mistral', 'test', 'image', 'audio', 'video', 
        'realtime', 'tts', 'embed', 'whisper', 'sora', 'vision', 'ocr', 
        'midjourney', 'computer-use', 'openaicontainers', 'deep-research',
        'deepresearch', 'qwq'
    ]
    
    if any(kw in v for kw in unwanted_keywords):
        return True
        
    if 'nano' in v or 'lite' in v:
        return True
        
    if 'mini' in v and 'minimax' not in v and 'mini-max' not in v:
        return True
    if handle == '/test':
        return True
    
    # Добавляем специфичные internal мусорные слова (например, zeliboba_models proxy, internal-model proxy, embedders, gigachat)
    if is_internal:
        if 'gigachat' in f or 'gigachat' in v:
            return True
        if 'embedder' in f or 'embedding' in v or 'embed' in v:
            return True
        if handle == 'unavailable':
            return True
        if 'proxy' in m.get('description', '').lower() or 'personal models' in m.get('description', '').lower():
            return True
        if 'zeliboba' in v or 'zeliboba' in f or 'yagpt' in f or 'yagpt' in v or 'yandex' in v or 'yandex' in f:
            return True
    else:
        if any(kw in f for kw in unwanted_keywords + ['gigachat']):
            return True
        
    unwanted_handles = ['images/generations', 'audio/speech', 'audio/transcriptions', '/realtime', '/embeddings', 'videos', '/search']
    if any(uh in handle for uh in unwanted_handles):
        return True
        
    if ('vl' in v and 'qwen' in v) or 'gpt-3.5' in v or 'oss' in v:
        return True
        
    if not is_internal:
        if v == 'deepseek-chat' or 'deepseek-r1-0528' in v:
            return True
        if 'kimi-k2-instruct' in v or v == 'moonshotai/kimi-k2':
            return True

    return False

url = "https://api.eliza.yandex.net/models"
headers = {"Authorization": "OAuth y1__xCapOORpdT-ARiuKyCqndUCMz-qrx_dO4vJCyeq3IL4tt2EuBk"}

try:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))

    models_list = data.get('models', data) if isinstance(data, dict) else data

    # 1. Собираем internal модели, применяя к ним фильтр (чтобы убрать test, embeddings, proxy и т.д.)
    internal_raw = [m for m in models_list if m.get('vendor') == 'internal']
    internal_filtered = [m for m in internal_raw if not is_unwanted(m, is_internal=True)]
    
    internal_filtered.sort(key=lambda x: x.get('model_version', '').lower())
    
    internal_data = {"models": internal_filtered} if isinstance(data, dict) else internal_filtered
    with open('internal_models.json', 'w', encoding='utf-8') as f:
        json.dump(internal_data, f, indent=4, ensure_ascii=False)

    # 2. Собираем основные внешние модели
    filtered_models = [m for m in models_list if not is_unwanted(m, is_internal=False)]

    family_tier_max_version = {}
    for m in filtered_models:
        family, tier, version = get_family_and_tier(m)
        key = (family, tier)
        if family != 'other':
            sp_versions = m.get('specific_versions', [])
            max_sp_v = version
            for sp_v in sp_versions:
                sp_v_parsed = parse_version(sp_v)
                if sp_v_parsed > max_sp_v:
                    max_sp_v = sp_v_parsed
            
            real_version = max(version, max_sp_v)
            
            if key not in family_tier_max_version or real_version > family_tier_max_version[key]:
                family_tier_max_version[key] = real_version

    final_models = []
    for m in filtered_models:
        family, tier, version = get_family_and_tier(m)
        key = (family, tier)
        
        sp_versions = m.get('specific_versions', [])
        max_sp_v = version
        for sp_v in sp_versions:
            sp_v_parsed = parse_version(sp_v)
            if sp_v_parsed > max_sp_v:
                max_sp_v = sp_v_parsed
        real_version = max(version, max_sp_v)
        
        if family == 'other':
            final_models.append(m)
        else:
            if real_version >= family_tier_max_version[key]:
                final_models.append(m)

    final_models.sort(key=lambda x: x.get('model_version', '').lower())

    final_data = {"models": final_models} if isinstance(data, dict) else final_models

    with open('models.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Успешно! {len(final_models)} топовых LLM в models.json:")
    for m in final_models:
        print(f"  - {m['model_version']}")
        
    print(f"\n✅ Успешно! {len(internal_filtered)} рабочих внутренних моделей в internal_models.json:")
    for m in internal_filtered:
        print(f"  - {m['model_version']}")
        
except Exception as e:
    print(f"❌ Ошибка: {e}")
