import time, random
from enum import IntFlag

# Genshin Impact Characters (Selection)
genshin_names = [
    "Aether", 
    "Lumine", 
    "Paimon", 
     
    "Phanes",
    "Istaroth",
    "Asmoday", 
    "Ronova",
    "Naberius",
    
    "Venti", 
    "Zhongli", 
    "RaidenEi", 
    "Nahida", 
    "Furina", 
    "Mavuika", 
    "Anastasya",
    
    "Neuvillette", 
    "Apep", 
    "Azhdaha", 
    "Dvalin", 
    "Nibelung", 
    "Xiuhcoatl",
    "Orobashi",
    
    "Varka",
    "JeanGunnhildr", 
    "Ningguang",
    "Yae Miko", 
    "Alhaitham", 
    "Arlecchino", 
    "Clorinde",
    "Dainsleif", 
    "Pierro", 
    
    "Diluc", 
    "Kaeya", 
    "Albedo",
    "HuTao",
    "Klee",
    "Xiao", 
    "Ganyu", 
    "Ayaka", 
    "Kazuha", 
    "Cyno", 
    "Wriothesley",
    "Navia", 
    "Kinich", 
    "Mualani", 
    "Childe"
]

# Honkai: Star Rail Characters (Selection)
hsr_names = [
    "Akivili", 
    "Pom-Pom", 
    "Welt", 
    "Himeko", 
    "DanHeng", 
    "March7th", 
    "Trailblazer",
    
    "Nanook", 
    "Lan", 
    "Nous", 
    "IX", 
    "Yaoshi", 
    "Xipe", 
    "Qlipoth", 
    
    "Elio", 
    "Kafka",
    "Blade", 
    "SilverWolf", 
    "Firefly-Lee", 
    "Diamond", 
    "Aventurine", 
    "Topaz", 
    "Jade",
    "Herta", 
    "RuanMei", 
    "Screwllum", 
    "DrRatio", 
    "JingYuan", 
    "FuXuan", 
    "Feixiao",
    "Huaiyan", 
    "Luocha", 
    "Sunday", 
    "Robin", 
    "BlackSwan", 
    "Acheron", 
    
    "Boothill",
    "Gepard", 
    "Bronya", 
    "Seele", 
    "Sparkle", 
    "Sampo", 
    "Tingyun", 
    "Yunli", 
    "Lingsha", 
    "Gallagher"
]

class ModelFlags(IntFlag):
    NONE = 0
    HAS_PCA = 1 << 0        # 1
    HAS_SYNTHETIC = 1 << 1  # 2
    HAS_OVERSAMP = 1 << 2   # 4
    IS_BALANCED = 1 << 3    # 8
    # Add more as needed: HAS_K_BEST = 1 << 4, etc.

def get_config_bitmask(has_pca: bool= False, has_synthetic: bool = False, has_oversamp: bool= False, is_balanced: bool= False) -> int:
    flags = ModelFlags.NONE
    if has_pca: flags |= ModelFlags.HAS_PCA
    if has_synthetic: flags |= ModelFlags.HAS_SYNTHETIC
    if has_oversamp: flags |= ModelFlags.HAS_OVERSAMP
    if is_balanced: flags |= ModelFlags.IS_BALANCED
    
    return int(flags)

def get_run_name(model_key: str, task: str, flags: int) -> str:
    timestamp = time.strftime("%H%M%d")
    local_random = random.Random(time.time_ns())
    selected_array = local_random.choice([genshin_names, hsr_names])
    if task == "neutralization": 
        task = "neutral"
        
    return f"{task}-{model_key}-{timestamp}-F{flags}-{local_random.choice(selected_array)}"