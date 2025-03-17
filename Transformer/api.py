from fastapi import FastAPI, HTTPException
import joblib
import torch
import numpy as np
import datetime
from train_transformer import FamiliarityTransformer  # Ensure correct model import
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import data_loader  # Assuming data_loader is the module that connects to your database

# Create FastAPI app
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

        past_words = pastWords.split(",")  # Split the input string into a list of words

        # ✅ Convert input words to indices
        try:
            encoded_words = [word_encoder.transform([word])[0] for word in past_words]  # Convert words to indices
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Unrecognized word(s): {e}")

        input_tensor = torch.tensor(encoded_words, dtype=torch.long).unsqueeze(0)  # Prepare input tensor for model

        with torch.no_grad():
            logits = model(input_tensor)  # Get logits from model predictions

            # ✅ Ensure correct logits shape
            if logits.dim() == 1:  
                logits = logits.unsqueeze(0)

            top_indices = torch.topk(logits, numWords, dim=-1).indices.squeeze().tolist()  # Get top N predictions

        # ✅ Ensure top_indices is a list
        if isinstance(top_indices, int):
            top_indices = [top_indices]

        # ✅ Filter out invalid indices
        max_index = len(word_encoder.classes_) - 1  # Maximum index should not exceed vocabulary size
        top_indices = [idx for idx in top_indices if 0 <= idx <= max_index]  # Remove invalid indices

        if not top_indices:
            raise HTTPException(status_code=500, detail="Model returned invalid indices out of range")

        # ✅ Convert indices back to words
        try:
            predicted_words = word_encoder.inverse_transform(np.array(top_indices))  # Convert indices back to words
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
                "createdAt": current_time  # Current time of the recommendation
            })

        return {"Recommended": recommended_words}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# **✅ Get least familiar words from the database**
def get_least_familiar_words(num_predictions=5):
    try:
        query = f"""
            SELECT TOP {num_predictions} Word
            FROM PersonalWords
            ORDER BY Familiarity ASC, NEWID();
        """
        df = pd.read_sql(query, data_loader.engine)  # Assuming data_loader.engine connects to your DB
        return df["Word"].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving least familiar words: {str(e)}")

# **✅ Endpoint to get least familiar words**
@app.get("/least_familiar_words")
def least_familiar(num_predictions: int = 5):
    try:
        least_familiar_words = get_least_familiar_words(num_predictions)
        return {"least_familiar_words": least_familiar_words}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving least familiar words: {str(e)}")

# **✅ Start FastAPI server**
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5001, reload=True)
