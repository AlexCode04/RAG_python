import openai
import os
import numpy as np
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

# Configuración de la API de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")  # Asegúrate de configurar esta variable de entorno


# Función para convertir PDFs a texto
def pdf_to_text(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Función para obtener embeddings usando OpenAI
def get_openai_embeddings(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Función para convertir documentos a vectores
def documents_to_vectors(documents):
    document_vectors = []
    for doc in documents:
        chunks = [doc[i:i+2048] for i in range(0, len(doc), 2048)]  # Dividir en fragmentos
        doc_vectors = [get_openai_embeddings(chunk) for chunk in chunks]
        doc_vector = np.mean(doc_vectors, axis=0)  # Promediar los vectores
        document_vectors.append(doc_vector)
    return document_vectors

# Función para encontrar documentos similares
def find_similar_documents(query, documents, filenames):
    query_vector = get_openai_embeddings(query)
    document_vectors = documents_to_vectors(documents)
    
    similarities = cosine_similarity([query_vector], document_vectors)[0]
    
    top_indices = np.argsort(similarities)[::-1]
    return [(filenames[i], similarities[i]) for i in top_indices]

# Función principal
def main():
    folder_path = 'docs/'
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    documents = []
    for filename in filenames:
        pdf_path = os.path.join(folder_path, filename)
        text = pdf_to_text(pdf_path)
        documents.append(text)
        print(f'Convertido {filename} a texto.')
    
    while True:
        query = input("Ingresa tu pregunta (o escribe 'salir' para terminar): ")
        if query.lower() == 'salir':
            break
        
        similar_docs = find_similar_documents(query, documents, filenames)
        
        if similar_docs:
            # Aquí se puede ajustar cómo se muestran los documentos similares
            print("Documento más similar:")
            print(f"Documento: {similar_docs[0][0]}")
            print(f"Similitud: {similar_docs[0][1]:.4f}")
        else:
            print("No se encontraron documentos similares.")
        
        # Generar una respuesta basada en el documento más similar
        most_similar_doc = similar_docs[0][0] if similar_docs else "No se encontraron documentos similares."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente útil."},
                {"role": "user", "content": f"Responde a la siguiente pregunta utilizando la información del documento: {most_similar_doc}\n\nPregunta: {query}"}
            ],
            max_tokens=150
        )
        print("Respuesta generada:")
        print(response.choices[0].message['content'].strip())

if __name__ == "__main__":
    main()
