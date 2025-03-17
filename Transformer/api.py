from fastapi import FastAPI, HTTPException
import joblib
import torch
import numpy as np
import datetime
from train_transformer import FamiliarityTransformer  # Ensure correct model import
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ✅ Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow React frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all request headers
)

# ✅ Load model & vocabulary encoder
word_encoder = joblib.load("word_encoder.pkl")
VOCAB_SIZE = len(word_encoder.classes_) + 1  # Ensure vocabulary size consistency
model = FamiliarityTransformer(vocab_size=VOCAB_SIZE)

# ✅ Initialize model and load state
model.load_state_dict(torch.load("familiarity_transformer.pth", map_location=torch.device("cpu")))
model.eval()  # Ensure inference mode

@app.get("/recommend")
def recommend(pastWords: str, numWords: int = 5):
    """
    **🔹 Recommendation API**
    - **pastWords**: User-provided words (comma-separated)
    - **numWords**: Number of words to recommend (default: 5)
    """
    try:
        if not pastWords:
            raise HTTPException(status_code=400, detail="pastWords parameter cannot be empty!")

        past_words = pastWords.split(",")

        # ✅ Convert input words to indices
        try:
            encoded_words = [word_encoder.transform([word])[0] for word in past_words]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Unrecognized word(s): {e}")

        input_tensor = torch.tensor(encoded_words, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            logits = model(input_tensor)

            # ✅ Ensure correct logits shape
            if logits.dim() == 1:  
                logits = logits.unsqueeze(0)

            top_indices = torch.topk(logits, numWords, dim=-1).indices.squeeze().tolist()

        # ✅ Ensure top_indices is a list
        if isinstance(top_indices, int):
            top_indices = [top_indices]

        # ✅ Filter out invalid indices
        max_index = len(word_encoder.classes_) - 1
        top_indices = [idx for idx in top_indices if 0 <= idx <= max_index]

        if not top_indices:
            raise HTTPException(status_code=500, detail="Model returned invalid indices out of range")

        # ✅ Convert indices back to words
        try:
            predicted_words = word_encoder.inverse_transform(np.array(top_indices))
        except ValueError:
            raise HTTPException(status_code=500, detail="Failed to convert indices to words, check training data")

        # ✅ Construct response with recommended words
        recommended_words = []
        current_time = datetime.datetime.utcnow().isoformat()  # Current UTC time

        for idx, word in enumerate(predicted_words):
            recommended_words.append({
                "id": idx + 1,  # Sequential ID
                "userId": 1,  # Default user ID (can be retrieved from DB)
                "word": word,
                "familiarity": 1,  # Default familiarity score
                "createdAt": current_time
            })

        return {"Recommended": recommended_words}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ✅ Start FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5001, reload=True)
