# web_scraper.py

import requests
from bs4 import BeautifulSoup
import re

# URL específica para aprofundar a busca (Indicadores sobre Ensino Superior)
url = "https://dadosabertos.mec.gov.br/indicadores-sobre-ensino-superior"

print(f"Tentando acessar e buscar downloads em: {url}")

try:
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    print(f"\nAcesso bem-sucedido a {url}.")
    print("\n--- Procurando por links de download de dados ---")

    found_download_links = {}

    # Palavras-chave para filtrar links que parecem ser de download de dados
    # Foco em termos como 'microdados', 'dados', 'arquivo', 'download' e anos.
    download_keywords = [
        'microdados', 'dados', 'download', 'arquivo', 'planilha',
        'censo', 'evasão', 'fluxo', 'indicadores',
        '2020', '2021', '2022', '2023', '2024' # Procurar por anos mais recentes
    ]

    # Regex para encontrar URLs que terminam com extensões de arquivo comuns
    file_extension_pattern = re.compile(r'\.(csv|zip|xlsx|txt|xls)$', re.IGNORECASE)

    # Procura por links que contenham as palavras-chave ou as extensões de arquivo
    for a_tag in soup.find_all('a', href=True):
        link_text = a_tag.get_text(strip=True)
        link_url = a_tag['href']

        # Tenta construir URL absoluta
        if not link_url.startswith('http'):
            link_url = requests.compat.urljoin(url, link_url)

        is_relevant_by_keyword = any(keyword in link_text.lower() for keyword in download_keywords) or \
                                any(keyword in link_url.lower() for keyword in download_keywords)

        is_download_link = bool(file_extension_pattern.search(link_url))

        if is_relevant_by_keyword or is_download_link:
            if link_url not in found_download_links:
                found_download_links[link_url] = link_text # Guarda a URL como chave e o texto como valor

    if found_download_links:
        print(f"Encontrados {len(found_download_links)} links de download potencialmente relevantes:")
        for link_url, link_text in found_download_links.items():
            print(f"  Texto: '{link_text}' -> URL: {link_url}")
    else:
        print("Nenhum link de download relevante encontrado nesta página.")
        print("Pode ser necessário explorar outras seções do portal ou refinar a busca.")

except requests.exceptions.RequestException as e:
    print(f"Erro ao acessar a URL: {e}")
    print("Verifique sua conexão com a internet ou se a URL está correta.")
except Exception as e:
    print(f"Ocorreu um erro: {e}")