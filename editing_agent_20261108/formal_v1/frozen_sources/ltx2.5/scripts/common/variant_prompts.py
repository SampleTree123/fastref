"""Reference-description changes shared by the three dataset variants.

Only known image-reference clauses change. History selection prefixes and
instructions to preserve unrelated attributes remain intact.
"""
from __future__ import annotations
import re
from prepare_datamake_manifest import require

def split_prefix(text, language, kind):
    """Only registered single-turn or explicit history-selection prefixes split."""
    if kind == 'single_turn':
        prefix = {'en': 'Using the provided source video, apply the following edit: ',
                  'zh': '请基于输入视频进行以下编辑：'}[language]
        require(text.startswith(prefix), 'Unknown single-turn prefix')
        return prefix, text[len(prefix):]
    require(kind in {'initial', 'latest', 'numbered', 'edit_event', 'single_feature'}, 'Unknown history selection kind')
    if kind in ('initial', 'latest'):
        return '', text
    marker = ':' if language == 'en' else '：'
    require(marker in text, 'Missing explicit history-selection separator')
    cut = text.index(marker) + 1
    while cut < len(text) and text[cut].isspace():
        cut += 1
    require(cut < len(text), 'Empty editing instruction')
    return text[:cut], text[cut:]

EN_PART = r'the (?:one-piece garment|top|bottoms|skirt|outerwear|footwear|accessory|gloves|wrap|bow|harness|external pockets)'

EN_PARTS = EN_PART + r'(?:(?:, | and |, and )' + EN_PART + r')*'

ZH_PARTS = r'(?:整身服装|上衣|下装|半身裙|外套|鞋子|配饰|手套|披肩|蝴蝶结|束带|外置口袋)(?:、(?:整身服装|上衣|下装|半身裙|外套|鞋子|配饰|手套|披肩|蝴蝶结|束带|外置口袋))*'

WHOLE_EN_SUFFIX = (
    'Remove any original clothing or footwear that is not part of this target outfit. Keep all non-clothing attributes unchanged.',
    'Keep only the clothing and footwear belonging to this target outfit, and leave all non-clothing aspects unchanged.',
    'This replaces the current outfit entirely, rather than adding items to it. Preserve everything else in the video.')

WHOLE_ZH_SUFFIX = (
    '移除原穿搭中不属于目标搭配的衣物或鞋子。保持所有非服装属性不变。',
    '仅保留属于目标搭配的衣物和鞋子，所有非服装内容保持不变。',
    '这是对当前穿搭的整体替换，而不是在原穿搭上添加单件。视频中的其他内容保持不变。')

def migrate_body(body, axis, mode, language):
    """Accept known template families only; preserve all non-reference clauses."""
    if axis == 'hair':
        if language == 'en':
            starts = ["Please restyle the person's hair according to the reference images.",
                      "restyle the person's hair according to the reference images.",
                      "Please change the person's hairstyle to match the reference images.",
                      "change the person's hairstyle to match the reference images.",
                      "I'd like to change the person's hairstyle to the one shown in the reference images."]
            ends = ["Match the shape and styling of the hair in the reference images, while retaining the current person's facial features, hair color, and clothing. Leave the rest of the video unchanged.",
                    "Use the reference images only for the hairstyle. Keep the person's identity, hair color, clothing, and all other aspects of the current video unchanged.",
                    "Only transfer the hairstyle from the reference images, not the reference person's facial features, hair color, or clothing. Preserve everything else in the current video."]
            require(any(body == a + ' ' + b for a in starts for b in ends), 'Unknown hair English template')
            return body.replace('reference images', 'reference image')
        starts = ['请按照参考图为人物更换发型。', '按照参考图为人物更换发型。', '请将人物的发型改为参考图中的发型。',
                  '将人物的发型改为参考图中的发型。', '我想将人物的发型更换为参考图中展示的发型。']
        ends = ['采用参考图中头发的形状和造型，同时保留当前人物的面部特征、发色和服装。视频中的其余内容保持不变。',
                '参考图仅用于确定发型。保持当前视频中人物的身份、发色、服装及其他内容不变。',
                '只采用参考图中的发型，不采用参考人物的面部特征、发色或服装。当前视频中的其他内容保持不变。']
        require(any(body == a + b for a in starts for b in ends), 'Unknown hair Chinese template')
        return body  # Appended layout sentence explicitly states front/back in one image.
    require(axis == 'wardrobe', 'Not a paired-image task')
    if mode == 'single_piece':
        if language == 'en':
            item = r"(?:one-piece garment|top|skirt|bottoms|outerwear)"
            patterns = [r"(?:Please r|r)eplace the person's " + item + r" with the item shown in the reference images\. Keep all other clothing, footwear, and non-clothing attributes unchanged\.",
                        r"I'd like to replace the person's " + item + r" with the item shown in the reference images\. Preserve the other clothing items, footwear, and all non-clothing aspects of the current video\.",
                        r"(?:Please c|c)hange only the person's " + item + r" to match the reference images\. Leave the rest of the outfit and everything else in the video unchanged\."]
            require(any(re.fullmatch(p, body) for p in patterns), 'Unknown single-piece English template')
            return body.replace('reference images', 'reference image')
        item = r'(?:整身服装|上衣|半身裙|下装|外套)'
        patterns = [r'(?:请)?将人物的' + item + r'替换为参考图中的单件。保持其他衣物、鞋子和非服装属性不变。',
                    r'我想将人物当前的' + item + r'换成参考图中的款式。保留当前视频中的其他衣物、鞋子及所有非服装内容。',
                    r'(?:请)?仅将人物的' + item + r'换成参考图中的款式。其余穿搭及视频中的其他内容保持不变。']
        require(any(re.fullmatch(p, body) for p in patterns), 'Unknown single-piece Chinese template')
        return body.replace('参考图中的', '参考图中以正面和背面展示的')
    require(mode == 'whole_outfit', 'Unknown wardrobe mode')
    if language == 'en':
        first, sep, tail = body.partition('. ')
        suffixes = [s + e for s in WHOLE_EN_SUFFIX for e in ('', ' Remove the current footwear and leave the person barefoot.', ' Wear all components of the reference outfit together as a complete set.')]
        require(sep and tail in suffixes, 'Unknown whole-outfit English preservation clauses')
        coord = r'the complete coordinated outfit, including (?:all of its components and the footwear|' + EN_PARTS + ')'
        patterns = [r"(?:Please r|r)eplace the person's entire outfit with the following items shown in the reference images: " + EN_PARTS,
                    r"(?:Please r|r)eplace the person's entire outfit with the complete coordinated outfit shown in the reference images, including (?:all of its components and the footwear|" + EN_PARTS + ')',
                    r"I'd like to replace the person's whole outfit with (?:" + EN_PARTS + '|' + coord + r'), as shown in the reference images',
                    r'Dress the person in the complete outfit shown in the reference images, consisting of ' + EN_PARTS,
                    r'Dress the person in the complete coordinated outfit shown in the reference images, including (?:all of its components and the footwear|' + EN_PARTS + ')']
        require(any(re.fullmatch(p, first) for p in patterns), f'Unknown whole-outfit English first sentence: {first}')
        first = first.replace('the following items shown in the reference images: ', 'the complete outfit shown from the front and back in the reference image, consisting of ')
        first = first.replace('shown in the reference images', 'shown from the front and back in the reference image')
        return first + sep + tail
    first, sep, tail = body.partition('。')
    suffixes = [s + e for s in WHOLE_ZH_SUFFIX for e in ('', '脱去当前的鞋子，让人物保持赤足。', ' 参考套装的所有组成部件需成套穿着。')]
    require(sep and tail in suffixes, 'Unknown whole-outfit Chinese preservation clauses')
    patterns = [r'(?:请)?将人物的整套穿搭替换为参考图中的以下单件：' + ZH_PARTS,
                r'(?:请)?将人物的整套穿搭替换为参考图中的完整配套套装，包括(?:全部组成部件及鞋子|' + ZH_PARTS + ')',
                r'我想将人物的整套穿搭更换为参考图中的(?:' + ZH_PARTS + r'|完整配套套装，包括(?:全部组成部件及鞋子|' + ZH_PARTS + '))',
                r'为人物换上参考图中由' + ZH_PARTS + '组成的完整穿搭',
                r'为人物换上参考图中的完整配套套装，包括(?:全部组成部件及鞋子|' + ZH_PARTS + ')']
    require(any(re.fullmatch(p, first) for p in patterns), f'Unknown whole-outfit Chinese first sentence: {first}')
    first = first.replace('参考图中的以下单件：', '参考图中以正面和背面展示的完整穿搭，包含以下部件：')
    if first.startswith('为人物换上参考图中由'):
        first = first.replace('为人物换上参考图中由', '为人物换上参考图中以正面和背面展示、由', 1)
    elif '以正面和背面展示' not in first:
        first = first.replace('参考图中', '参考图中以正面和背面展示的', 1).replace('展示的的', '展示的')
    return first + sep + tail

def migrate_prompt(prompt, axis, mode, aspect, selection_kind):
    result = {}
    suffix = {'en': ' The reference image shows the front view on the left and the back view on the right.' if aspect == 1 else ' The reference image shows the front view above and the back view below.',
              'zh': '参考图左侧为正面，右侧为背面。' if aspect == 1 else '参考图上方为正面，下方为背面。'}
    for lang in ('en', 'zh'):
        prefix, body = split_prefix(prompt[lang], lang, selection_kind)
        result[lang] = prefix + migrate_body(body, axis, mode, lang) + suffix[lang]
        require(result[lang].startswith(prefix), 'Historical selection prefix changed')
    return result
