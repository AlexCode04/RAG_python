import os
import faiss
import numpy as np
import spacy
import fitz  # PyMuPDF

# Cargar el modelo de spaCy
nlp = spacy.load('en_core_web_lg')

# Cargar y procesar documentos PDF
def load_documents(directory):
    documents = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            with fitz.open(os.path.join(directory, filename)) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                documents.append(text)
                filenames.append(filename)
    return documents, filenames

# Convertir documentos a vectores usando spaCy
def text_to_vectors(texts):
    vectors = []
    for text in texts:
        doc = nlp(text)
        vectors.append(doc.vector)
    return np.array(vectors)

# Normalizar los vectores
def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# Crear un índice FAISS
def create_faiss_index(vectors):
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Índice para producto interno (dot product)
    index.add(vectors)
    return index

# Consultar el índice FAISS
def query_index(index, vector, k=5):
    distances, indices = index.search(np.array([vector]), k)
    return distances, indices

# Encontrar un fragmento relevante en el documento
def find_relevant_snippet(document, query, nlp):
    doc = nlp(document)
    query_doc = nlp(query)
    
    # Verificar si hay oraciones y si los vectores no son vacíos
    if not list(doc.sents) or query_doc.vector_norm == 0:
        return "No se encontró un fragmento relevante."
    
    similarities = []
    for sent in doc.sents:
        if sent.vector_norm != 0:  # Asegurarse de que el vector no esté vacío
            sim = query_doc.similarity(sent)
            similarities.append(sim)
    
    if not similarities:  # Verificar si la lista de similitudes está vacía
        return "No se encontró un fragmento relevante."
    
    most_similar_sentence = max(similarities)
    most_similar_index = similarities.index(most_similar_sentence)
    return list(doc.sents)[most_similar_index].text

def main():
    # Cargar documentos y convertirlos a vectores
    directory = './docs'
    documents, filenames = load_documents(directory)
    
    # Verificar si se cargaron documentos
    if len(documents) == 0:
        print("No se encontraron documentos en el directorio.")
        return

    vectors = text_to_vectors(documents)
    
    # Verificar si se generaron vectores
    if vectors.size == 0:
        print("No se encontraron vectores. Saliendo.")
        return
    
    vectors = normalize_vectors(vectors)
    print(f"Shape of vectors: {vectors.shape}")

    # Crear el índice FAISS
    index = create_faiss_index(vectors)

    while True:
        # Realizar consulta
        query = input("Ingresa tu pregunta (o escribe 'salir' para terminar): ")
        
        if query.lower() == 'salir':
            print("Saliendo del programa.")
            break
        
        query_doc = nlp(query)
        
        # Verificar si el vector de consulta es un vector vacío
        if query_doc.vector_norm == 0:
            print(f"No se encontraron documentos relevantes para la consulta '{query}'.")
            continue
        
        query_vector = query_doc.vector / query_doc.vector_norm  # Normalizar el vector de consulta

        # Pedir al usuario la cantidad de resultados deseados
        try:
            num_results = int(input("¿Cuántos resultados deseas mostrar? "))
            if num_results <= 0:
                raise ValueError("El número de resultados debe ser un entero positivo.")
        except ValueError as e:
            print(f"Entrada no válida: {e}. Mostrando 5 resultados por defecto.")
            num_results = 5

        distances, indices = query_index(index, query_vector, k=num_results)
        
        # Definir un umbral de similitud para filtrar resultados irrelevantes
        threshold = 0.1  # Ajusta este valor según tus necesidades

        print("Top documents:")
        found_relevant = False
        for dist, i in zip(distances[0], indices[0]):
            if i == -1 or dist == 3.4028234663852886e+38:  # Evitar índices no válidos
                continue
            similarity = dist  # En FAISS con producto interno, la distancia es la similitud
            if similarity > threshold:
                found_relevant = True
                snippet = find_relevant_snippet(documents[i], query, nlp)
                print((60*"-")+'\n')
                print(f"Document: {filenames[i]}")
                print(f"Similarity: {similarity}")
                print(f"Relevant Snippet: {snippet}")
                print((60*"-")+'\n')
        
        if not found_relevant:
            print(f"No se encontraron documentos relevantes para la consulta '{query}'.")

if __name__ == "__main__":
    main()
