import os
import re

def try_decode(file_path):
    encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-16']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                content = f.read()
                # If we get here, it decoded. Let's check for replacement characters or weird artifacts.
                if '\x00' in content and enc != 'utf-16':
                    continue
                return content, enc
        except Exception:
            continue
    # Fallback to binary read and ignore errors
    with open(file_path, 'rb') as f:
        return f.read().decode('utf-8', errors='ignore'), 'utf-8-ignored'

def clean_text(text):
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Merge hyphenated words across line breaks (e.g. Douto-\n\nrado -> Doutorado)
    text = re.sub(r'(\w+)-\s*\n+\s*(\w+)', r'\1\2', text)
    
    # Fix common encoding artifacts
    fixes = {
        'a': 'ça',
        'o': 'ão',
        '': 'ç',  # general fallback if needed, but let's be careful
    }
    
    # Let's check if there are specific corrupted words
    # e.g., Apresentaao -> Apresentação, Coordenaao -> Coordenação
    text = re.sub(r'Apresenta[^\w\s]?ao', 'Apresentação', text)
    text = re.sub(r'Coordena[^\w\s]?ao', 'Coordenação', text)
    text = re.sub(r'Selec[^\w\s]?ao', 'Seleção', text)
    text = re.sub(r'selec[^\w\s]?ao', 'seleção', text)
    text = re.sub(r'pos-gradua[^\w\s]?ao', 'pós-graduação', text)
    text = re.sub(r'P[^\w\s]?os-Gradua[^\w\s]?ao', 'Pós-Graduação', text)
    text = re.sub(r'p[^\w\s]?os-gradua[^\w\s]?ao', 'pós-graduação', text)
    text = re.sub(r'Gradua[^\w\s]?ao', 'Graduação', text)
    text = re.sub(r'gradua[^\w\s]?ao', 'graduação', text)
    text = re.sub(r'avalia[^\w\s]?ao', 'avaliação', text)
    text = re.sub(r'publica[^\w\s]?ao', 'publicação', text)
    text = re.sub(r'inscri[^\w\s]?ao', 'inscrição', text)
    text = re.sub(r'Inscri[^\w\s]?ao', 'Inscrição', text)
    text = re.sub(r'inscri[^\w\s]?oes', 'inscrições', text)
    text = re.sub(r'Inscri[^\w\s]?oes', 'Inscrições', text)
    text = re.sub(r'dura[^\w\s]?ao', 'duração', text)
    text = re.sub(r'Dura[^\w\s]?ao', 'Duração', text)
    text = re.sub(r'Composi[^\w\s]?ao', 'Composição', text)
    text = re.sub(r'composi[^\w\s]?ao', 'composição', text)
    
    return text

def sliding_window_segmentation(text, max_words=250, overlap=100):
    words = text.split()
    if len(words) <= max_words:
        return [text]
        
    segments = []
    step = max_words - overlap
    i = 0
    while i < len(words):
        window_words = words[i : i + max_words]
        window_text = " ".join(window_words)
        segments.append(window_text)
        if i + max_words >= len(words):
            break
        i += step
    return segments

def segment_text(file_name, text):
    # Depending on the file size and structure, we split it into logical chunks.
    # We want each chunk to represent a clean, self-contained context.
    raw_segments = []
    
    # 1. Special handling for Regulamentos.txt (which is very large)
    if 'Regulamentos.txt' in file_name:
        # Split by chapters or major sections
        chapters = re.split(r'(CAPÍTULO\s+[I|V|X|L|C|D|M]+|Capítulo\s+[I|V|X|L|C|D|M]+)', text)
        
        # The first segment is intro regulamentos
        intro = chapters[0].strip()
        if intro:
            raw_segments.append({
                "subsection": "Introdução aos Regulamentos",
                "context": intro
            })
            
        i = 1
        while i < len(chapters):
            chap_header = chapters[i].strip()
            chap_content = chapters[i+1].strip() if i+1 < len(chapters) else ""
            
            # Let's combine them
            full_chap_text = chap_header + "\n" + chap_content
            
            # If the chapter is very long, we can further split by Section or by a group of Articles
            if len(full_chap_text) > 3000:
                # Split by "Seção" or "Art. "
                sections = re.split(r'(Seção\s+[I|V|X|L|C|D|M]+|Art\.\s+\d+o?)', full_chap_text)
                sub_intro = sections[0].strip()
                if sub_intro:
                    raw_segments.append({
                        "subsection": chap_header,
                        "context": sub_intro
                    })
                
                j = 1
                while j < len(sections):
                    sec_header = sections[j].strip()
                    sec_content = sections[j+1].strip() if j+1 < len(sections) else ""
                    sub_text = sec_header + " " + sec_content
                    
                    if len(sub_text) > 2500:
                        # Split by "Art."
                        articles = re.split(r'(Art\.\s+\d+o?)', sub_text)
                        art_intro = articles[0].strip()
                        if art_intro:
                            raw_segments.append({
                                "subsection": f"{chap_header} - {sec_header}",
                                "context": art_intro
                            })
                        k = 1
                        while k < len(articles):
                            art_header = articles[k].strip()
                            art_content = articles[k+1].strip() if k+1 < len(articles) else ""
                            art_text = art_header + " " + art_content
                            raw_segments.append({
                                "subsection": f"{chap_header} - {sec_header} - {art_header}",
                                "context": art_text.strip()
                            })
                            k += 2
                    else:
                        raw_segments.append({
                            "subsection": f"{chap_header} - {sec_header}",
                            "context": sub_text.strip()
                        })
                    j += 2
            else:
                raw_segments.append({
                    "subsection": chap_header,
                    "context": full_chap_text.strip()
                })
            i += 2
            
    else:
        # 2. General files
        if len(text) < 1500:
            raw_segments.append({"subsection": "Geral", "context": text.strip()})
        elif "-------------------" in text:
            parts = re.split(r'-{10,}', text)
            for idx, part in enumerate(parts):
                p = part.strip()
                if p:
                    raw_segments.append({
                        "subsection": f"Parte {idx+1}",
                        "context": p
                    })
        else:
            paragraphs = text.split('\n\n')
            current_segment = []
            current_len = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if current_len + len(para) > 1500 and current_segment:
                    raw_segments.append({
                        "subsection": "Geral",
                        "context": "\n\n".join(current_segment)
                    })
                    current_segment = [para]
                    current_len = len(para)
                else:
                    current_segment.append(para)
                    current_len += len(para) + 2
                    
            if current_segment:
                raw_segments.append({
                    "subsection": "Geral",
                    "context": "\n\n".join(current_segment)
                })
                
    # Post-procesamiento: Aplicar Ventana Deslizante para cualquier segmento que exceda las 250 palabras
    final_segments = []
    for s in raw_segments:
        ctx = s["context"].strip()
        words = ctx.split()
        if len(words) > 250:
            windows = sliding_window_segmentation(ctx, max_words=250, overlap=100)
            for w_idx, win in enumerate(windows):
                final_segments.append({
                    "subsection": f"{s['subsection']} - Parte {w_idx+1}",
                    "context": win.strip()
                })
        else:
            final_segments.append(s)
            
    return [s for s in final_segments if len(s["context"].strip()) > 30]

def main():
    info_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "documents")
    print("Iniciando preprocesamiento y análisis de documentos...")
    
    for root, dirs, files in os.walk(info_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                content, enc = try_decode(file_path)
                cleaned = clean_text(content)
                segments = segment_text(file, cleaned)
                
                rel_path = os.path.relpath(file_path, info_dir)
                print(f"Archivo: {rel_path} | Codificación detectada: {enc} | Segmentos extraídos: {len(segments)}")

if __name__ == "__main__":
    main()
