# RAG System with Gemini and FAISS

This Jupyter notebook implements a Retrieval-Augmented Generation (RAG) system using the Gemini language model and FAISS vector store. Here's a brief overview:

## Overview

- **Purpose**: To create a conversational AI system that can answer questions based on a given document.
- **Components**:
  - **Gemini**: A language model for generating responses.
  - **FAISS**: A vector store for efficient document retrieval.
  - **LangChain**: A framework for building applications with language models.

## Setup

1. **Environment Variables**: Ensure you have the `GEMINI_API_KEY` set in your environment.
2. **Dependencies**: Install required libraries in requirements.txt


## Usage

1. **Download and Process Document**: The system downloads a PDF document, processes it into chunks, and stores these chunks in a FAISS vector store.
2. **Create RAG Chain**: A conversational retrieval chain is created using the Gemini model and the FAISS vector store.
3. **Query the System**: You can ask questions, and the system will retrieve relevant information from the document to answer them.

## Running the System

- **Build the RAG System**: Run the `build_rag_system` function with a document URL to set up the system.
- **Test Queries**: Use the `query_rag` function to ask questions and get answers.

## Example Queries

- Can you explain the concept of 'combat' in simple terms?
- How does one win a game of Magic: The Gathering?
- What is haste?

## Notes

- If the question is not related to the context, the system will respond with "I don't know."
- The document used in this example is the Magic: The Gathering Comprehensive Rules.


