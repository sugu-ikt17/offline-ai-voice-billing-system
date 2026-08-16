"""Aggregates all v1 routers into a single APIRouter."""

from fastapi import APIRouter

from app.presentation.api.v1 import (
    bill_routes,
    menu_routes,
    order_routes,
    speech_routes,
    voice_routes,
)

api_router = APIRouter()
api_router.include_router(menu_routes.router)
api_router.include_router(speech_routes.router)   # POST /api/v1/speech/transcribe
api_router.include_router(voice_routes.router)    # POST /api/v1/voice/transcribe (legacy)
api_router.include_router(order_routes.router)
api_router.include_router(bill_routes.router)
