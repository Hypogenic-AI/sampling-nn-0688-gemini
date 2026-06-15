import arxiv
import os
import requests

papers = [
    {"id": "1905.10761", "title": "ProbAct_Probabilistic_Activation_Function"},
    {"id": "1603.00391", "title": "Noisy_Activation_Functions"},
    {"id": "2007.10412", "title": "Randomized_Automatic_Differentiation"},
    {"id": "1505.05424", "title": "Weight_Uncertainty_Neural_Networks"},
    {"id": "1312.6114", "title": "VAE_Original_Paper"},
    {"id": "2601.06441", "title": "FlexAct_Why_Learn_When_You_Can_Pick"},
]

client = arxiv.Client()
os.makedirs("papers", exist_ok=True)

for paper_info in papers:
    try:
        search = arxiv.Search(id_list=[paper_info["id"]])
        paper = next(client.results(search), None)
        if paper:
            filename = f"{paper_info['id']}_{paper_info['title']}.pdf"
            filepath = os.path.join("papers", filename)
            if not os.path.exists(filepath):
                print(f"Downloading {paper.title} from {paper.pdf_url}...")
                response = requests.get(paper.pdf_url)
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Saved to {filepath}")
            else:
                print(f"Skipping {filename}, already exists.")
        else:
            print(f"Could not find paper with ID {paper_info['id']}")
    except Exception as e:
        print(f"Error downloading {paper_info['id']}: {e}")
