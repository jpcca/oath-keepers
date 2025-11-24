from typing import List
from pydantic import BaseModel, Field, model_validator

class AgeBin(BaseModel):
    bin_start: int = Field(..., description="Inclusive start of the age bin")
    bin_end: int = Field(..., description="Exclusive end of the age bin")
    p: float = Field(..., description="Probability for this age bin, between 0 and 1")

class AgeDistribution(BaseModel):
    bins: List[AgeBin] = Field(..., description="List of age bins covering 0 to 100")

    @model_validator(mode='after')
    def validate_distribution(self) -> 'AgeDistribution':
        # Check if probabilities sum to 1.0 (with tolerance)
        total_prob = sum(b.p for b in self.bins)
        
        # If total probability is close to 1.0 (e.g., 0.9 to 1.1), normalize it.
        # LLMs often make small arithmetic errors.
        if 0.9 <= total_prob <= 1.1:
            if abs(total_prob - 1.0) > 1e-6:
                for b in self.bins:
                    b.p /= total_prob
        else:
            raise ValueError(f"Total probability must be close to 1.0, got {total_prob:.4f}")
        
        # Check if bins cover 0-100 (simplified check: just check start/end of first/last and continuity)
        # This is a basic check, can be more robust if needed.
        sorted_bins = sorted(self.bins, key=lambda x: x.bin_start)
        if not sorted_bins:
             raise ValueError("Bins cannot be empty")
             
        if sorted_bins[0].bin_start != 0:
             raise ValueError("Bins must start at 0")
             
        # Check continuity
        for i in range(len(sorted_bins) - 1):
            if sorted_bins[i].bin_end != sorted_bins[i+1].bin_start:
                 raise ValueError(f"Gap or overlap between bins: [{sorted_bins[i].bin_start},{sorted_bins[i].bin_end}) and [{sorted_bins[i+1].bin_start},{sorted_bins[i+1].bin_end})")
                 
        if sorted_bins[-1].bin_end != 100:
             raise ValueError("Bins must end at 100")
             
        return self
