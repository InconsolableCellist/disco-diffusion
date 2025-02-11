import json, os, random, numpy

settings = [
    "memes2.json"
]

artists_favorite = [
    "Tom Bagshaw", "Miyazaki", "Caspar David Friedrich",
    "Quentin Mabille", "Ivan Konstantinovich Aivazovsky","Makoto Shinkai", "Eugene Korolev"
    "Moebius", "laloux", "Carlos Ortega Elizalde", "Rafał Olbiński"
]

artists_anime = [
    "makoto Shinkai", "Hiroshi Yoshida", "Sin jong hun", "Miyazaki", "Ivan Bilibin",  "Studio Ghibli"
]

artists_illustration = [
    "fernando chamarelli", "matt mills", "Martin Johnson Heade"
]
artists = [
    "Leegan Koo", "Simon Stålenhag", "James Jean", "Boris Pelcer", "eugene korolev", "Olga Kim", "ismail inceoglu",
    "ulysse verhassel", "francisco martin", "henry wong", "axel sauerwald", "alexandr poda", "ingram schell",
    "mael ollivier henry", "ayan nag", "Ben Bauchau", "Syd Mead", "dariusz zawadzki", "huleeb", "frank mccarty",
    "Oscar Berninghaus", "Jim C Norton", "Thomas Cole", "makoto Shinkai", "Hiroshi Yoshida", "Sin jong hun", "Miyazaki",
    "Ivan Bilibin", "Studio Ghibli", "Cedric Peyravernay", "Peter Mohrbacher", "Tom Bagshaw", "Karol Bak",
    "Steve Argyle", "Hethe Srodawa", "Alena Aenami", "Philipp A. Urlich", "Tyler Edlin", "Anthony Efthekary",
    "marco gorlei", "cody foreman", "pavel vophira", "john sweeney", "roma likholob", "gilles ketting", "pete amachree",
    "sung choi", "victor hugo harmatiuk", "Min Guen", "Paul Chadeisson", "Jaime Jasso", "Jonathan Berube",
    "Steven Cormann", "Jaime Jasso", "Thu Berchs", "Terraform Studios", "Daniel Dana", "Erik van Helvoirt", "Luc Begin",
    "Sanhanat Suwanwised", "Hossein Diba", "Raf Grassetti", "Hadi Karimi", "Frank Tzeng", "Şefki Ibrahim",
    "Victor Hugo", "James Gurney", "Mark Simonetti", "Katsuhiro Otomo", "Yoshitaka Amano", "vance kovac",
    "Arnold Böcklin", "Fitz Hugh Lane", "Koen Wijffelaars", "Ian McQue", "Jan Urschel", "Craig Mullins",
    "Ivan Konstantinovich Aivazovsky", "nadia hurianova", "Luke berliner", "quentin mabille", "Liang Mark",
    "andrei riabovitchev", "john burton" "giorgi simeonov", "mathias zamecki", "Lloyd Allan", "John Stephens",
    "Peder Severin Krøyer", "albert Bierstadt", "Greg Rutkowski", "Caspar David Friedrich", "John Berkey",
    "Asher Brown Durand", "Nadezda Petrovic", "William Bradford", "Edvard Munch", "Vincent Van Gogh", "Gustav Klimt",
    "Claude Monet", "Eugene Delacroix", "Hieronymous Bosch", "Egon Schiele", "Henri de Toulouse-Lautrec",
    "Jean-Honoré Fragonard", "Jean-Michel Basquiat", "Android Jones", "Justin Totemical", "Kirsten Zirngibl",
    "Alex Grey", "andy gilmore", "jonathan solter", "ben ridgway", "kent baltutat", "Jake amason",
    "brian scott hampton", "CT Nelson", "Leszek Kostuj", "Rae Klein", "Yuri Yakovenko", "Eugenia Loli", "Dorris Vooijs",
    "Celine Ranger", "fernando chamarelli", "matt mills", "Martin Johnson Heade", "James Paick", "H R Giger",
    "John Harris" "Frederico Pelat", "JC Jongwon Park ", "Zeen Chin", "Ellen Jewett", "Nadezda sokolova", "Panamarenko",
    "kris kuksi", "Peter Elson", "John Berkey", "Roger Dean", "Pascal Blanché", "Mœbius", "Moebius", "Jean Giraud",
    "Sydney Jay Mead", "Kilian Eng", "Geof Darrow", "frank frazetta", "Bastien Lecouffe-Deharme", "Tuomas Korpi",
    "Daryl Mandryk", "Anato Finnstark", "noah bradley", "Aria Fawn", "Jeremy fenske", "leonid koliagin",
    "Sylvain Sarrailh", "Adam Burke", "Caelen Stokkermans", "Vex Van Hillik ", "Wiesław Wałkuski", "Akiya Kageichi",
    "Beeple", "Mike Winkelmann", "Filip Hodas", "Farid Ghanbari", "Billy Bogatzoglus", "Anwar Mostafa",
    "Stuart Lippincott", "Ari Weinkle", "Aleksandr Kuskov", "Gesy Bekeyei", "Andy Goralkcyzk", "Wes Anderson",
    "Liam Wong", "Bastien Grivet", "Stephen Shore", "Alphons Mucha", "Juras Rodionovas", "Soa Lee", "Ilya Kuvshinov",
    "Lucas Mendonça" "josan gonzalez", "Archan Nair", "Viktoria Gavrilenko", "Everett Kinstler", "KwangHo Shin",
    "Abdelrahman Taymour"]

prompts = []
common_prompts = []
defaults = {}
overrides = {}
presets = {}

for x in range(1, 50):
    with open('prompts.json', 'r') as f:
        data = json.load(f)
        prompts = data['memes']
        # prompts = data['reddit_posts']
        common_prompts = data['common_prompts']
        if 'defaults' in data:
            defaults = data['defaults']
        if 'presets' in data:
            presets = data['presets']
    for prompt in prompts:
        p = {}
        with open(settings[0], 'r') as f:
            p = json.load(f)
        p['init_image'] = f'init_images\\init{random.randint(1,33)}.jpg'
        cgs = abs(numpy.random.normal(loc=0, scale=1, size=1))
        cgs *= 30000
        p['clip_guidance_scale'] = int(cgs)
        p['tv_scale'] = random.randint(200, 700)
        if random.randint(0, 1) == 0:
            p['use_secondary_model'] = False
        else:
            p['use_secondary_model'] = True

        for key, value in defaults.items():
            p[key] = value
            print(f'setting default {key} to {value}')

        if 'preset' in prompt:
            preset = prompt['preset']
            if preset in presets:
                print(f"Prompt requested preset {preset}, which we found")
                for key, value in presets[preset]['defaults'].items():
                    print(f"\tPreset {preset} is overriding default {key} with {value}")
                    p[key] = value
            else:
                print(f"The prompt requested preset {preset}, but didn't find it")


        if 'overrides' in prompt:
            for key, value in prompt['overrides'].items():
                if key == "init_image" and isinstance(value, list):
                    value = random.choice(value)
                p[key] = value
                print(f'overriding {key} to {value}')

        if not isinstance(prompt['prompt'], list):
            prompt['prompt'] = [prompt['prompt']]

        if 'add_artist' in prompt and prompt['add_artist'] == True:
            if 'artist_category' in prompt:
                if prompt['artist_category'] == "anime":
                    prompt['prompt'].append(f'by {random.choice(artists_anime)}:4')
                if prompt['artist_category'] == "illustration":
                    prompt['prompt'].append(f'by {random.choice(artists_illustration)}:4')
                if prompt['artist_category'] == "favorites":
                    prompt['prompt'].append(f'by {random.choice(artists_favorite)}:4')
            else:
                prompt['prompt'].append(f'by {random.choice(artists)}:4')

        for c in common_prompts:
            prompt['prompt'].append(c)
        p['text_prompts']['0'] = prompt['prompt']

        with open(settings[0], 'w') as f:
            json.dump(p, f)
        os.system("python disco.py \"" + settings[0] + "\"")


