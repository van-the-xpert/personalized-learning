from langchain_core.pydantic_v1 import BaseModel, Field

# Output Parser
class LessonContent(BaseModel):
    summary: str = Field(description="The summary of the content briefly which can be referred to as lesson contents")
    topics: str = Field(description="Valuable topics from the summary at most 3, one topic per line")
