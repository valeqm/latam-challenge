from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import pandas as pd
from challenge.model import DelayModel
from typing import List, Union

app = FastAPI()
modelo = DelayModel()

operas_validos = [
    "Aerolineas Argentinas",
    "Aeromexico",
    "Air Canada",
    "Air France",
    "Alitalia",
    "American Airlines",
    "Austral",
    "Avianca",
    "British Airways",
    "Copa Air",
    "Delta Air",
    "Gol Trans",
    "Grupo LATAM",
    "Iberia",
    "JetSmart SPA",
    "K.L.M.",
    "Lacsa",
    "Latin American Wings",
    "Oceanair Linhas Aereas",
    "Plus Ultra Lineas Aereas",
    "Qantas Airways",
    "Sky Airline",
    "United Airlines"
]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()},
    )

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data: dict) -> dict:
    try:
        global modelo  
        
        flights = data.get("flights", [])
        predictions_list=[]
        
        if not flights:
            raise HTTPException(status_code=400, detail="No flights data provided.")

        for flight in flights:
            mes = flight.get("MES", None)
            tipovuelo = flight.get("TIPOVUELO", None)
            opera = flight.get("OPERA", None)
            
            if (
                mes is None or not (1 <= mes <= 12) or
                tipovuelo not in ["I", "N"] or
                opera is None or opera not in operas_validos
            ):
                raise HTTPException(status_code=400, detail="Invalid flight data.")


            df = pd.DataFrame([flight])
            data_process = modelo.preprocess(df)
            prediction = modelo.predict(data_process)
            predictions_list.append(prediction[0]) 

        return {"predict": predictions_list}
    

    
    except HTTPException as e:
        print(f"HTTP Exception: {e.status_code}, Detail: {e.detail}")
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        print(f"Exception: {str(e)}")
        return {"error": str(e)}
    
