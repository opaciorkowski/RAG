import json
from pathlib import Path
from langchain.prompts import PromptTemplate

class PromptManager:
    def __init__(self, prompt_file: str = None):
      if prompt_file is None:
          prompt_file = Path(__file__).resolve().parent / "prompt_templates.json"
      self.prompt_file = Path(prompt_file)
      self.prompts = self._load_prompts()

    def _load_prompts(self):
        if not self.prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file}")
        with open(self.prompt_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_prompt(self, prompt_type: str, role: str = "AI Assistant") -> PromptTemplate:
        template = self.prompts.get(prompt_type)
        if not template:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        if "{role}" in template:
            template = template.replace("{role}", role)
        return PromptTemplate.from_template(template)

    def rewrite_query(self, original_query: str, llm, retriever) -> str:
        relevant_docs = retriever.get_relevant_documents(original_query)
        context = "\n---\n".join(doc.page_content for doc in relevant_docs[:3]) if relevant_docs else ""
        template = self.prompts["rewrite"]
        prompt = PromptTemplate.from_template(template)
        chain = prompt | llm
        response = chain.invoke({"question": original_query, "context": context})
        return response.content if hasattr(response, "content") else str(response)

