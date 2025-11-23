from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

class ProtoConceptExtract(BaseModel):
    reasoning: Optional[str] = Field(description="string describing reasoning for which keyphrases to include")
    keyphrases: List[str] = Field(description="string of keyphrases that are highly related to each other")

class ProtoConceptExtractGrouped(BaseModel):
    reasoning: Optional[str] = Field(description="string describing reasoning for which keyphrases to include")
    keyphrases: List[str] = Field(description="string of keyphrases for each bird")

class PriorCandidate(BaseModel):
    candidate_id: int = Field(description="Candidate concept ID")
    reasoning: Optional[str] = Field(description="Reasoning for the prior assigned to this candidate")
    prior: float = Field(description="Prior assigned to this candidate concept")

class PriorResponse(BaseModel):
    candidate_priors: List[PriorCandidate] = Field(description="List of candidate concepts and their priors")

    def fill_candidate_concept_dicts(self, candidate_concept_dicts):
        prior_dict = {}
        for prior_candidate in self.candidate_priors:
            prior_dict[prior_candidate.candidate_id] = prior_candidate.prior

        for idx, concept_dict in enumerate(candidate_concept_dicts):
            if idx in prior_dict:
                concept_dict['prior'] = prior_dict[idx]
        return candidate_concept_dicts

class CandidateConcept(BaseModel):
    concept: str = Field(description="Concept defined as a yes/no question")
    words: List[str] = Field(description="Words that are synonyms or antonyms")

class CandidateConcepts(BaseModel):
    reasoning: Optional[str] = Field(description="Reasoning through each one of the attributes")
    concepts: List[CandidateConcept] = Field(description="List of candidate concepts")

    def to_dicts(self, default_prior:float=1):
        all_concept_dicts = []
        for concept in self.concepts:
            all_concept_dicts.append({
                "concept": concept.concept,
                "words": concept.words,
                "prior": default_prior,
            })
        return all_concept_dicts

class ExtractResponse(BaseModel):
    question: int = Field(..., description="question number")
    reasoning: Optional[str] = Field(description="reasoning")
    answer: float = Field(..., description="binary answer, 1=yes, 0=no, probability if unsure")

    @field_validator('question', mode="before")
    def ensure_question(cls, value):
        return int(value)

class ExtractResponseList(BaseModel):
    reasoning: Optional[str] = Field(description="reasoning")
    extractions: List[ExtractResponse] = Field(..., description="List of extractions")

class GroupedExtractResponses(BaseModel):
    all_extractions: List[ExtractResponseList] = Field(..., description="List of extractions")
