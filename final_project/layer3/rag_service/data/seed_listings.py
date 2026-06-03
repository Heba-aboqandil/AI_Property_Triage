"""
Pre-populate ChromaDB with 25 synthetic real estate property listings.

Run once before starting the RAG service:
    python data/seed_listings.py

The script is idempotent: re-running it clears and recreates the collection.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = "property_listings"

LISTINGS = [
    {
        "id": "listing_01",
        "title": "3-Bedroom Apartment in Tel Aviv - Sea View",
        "text": (
            "Spacious 3-bedroom apartment located on the 8th floor in the heart of Tel Aviv. "
            "Features an open-plan living area, renovated kitchen, two full bathrooms, and a large "
            "balcony with panoramic sea views. Building amenities include a gym, underground parking, "
            "and a 24-hour doorman. Listed at 4,200,000 ILS."
        ),
        "type": "apartment", "city": "Tel Aviv", "rooms": 3, "price_ils": 4200000,
    },
    {
        "id": "listing_02",
        "title": "4-Bedroom Villa in Herzliya Pituah",
        "text": (
            "Detached villa with private pool in the prestigious Herzliya Pituah neighbourhood. "
            "Four bedrooms, three bathrooms, a study, and a large garden. Recently renovated with "
            "smart-home automation. Asking price 9,800,000 ILS. Walking distance to the beach."
        ),
        "type": "villa", "city": "Herzliya", "rooms": 4, "price_ils": 9800000,
    },
    {
        "id": "listing_03",
        "title": "Studio Apartment in Jerusalem - Old City View",
        "text": (
            "Compact studio apartment in central Jerusalem with a stunning view of the Old City walls. "
            "Fully furnished, air-conditioned, and suitable for investment or personal use. "
            "50 sqm. Asking 1,150,000 ILS."
        ),
        "type": "apartment", "city": "Jerusalem", "rooms": 1, "price_ils": 1150000,
    },
    {
        "id": "listing_04",
        "title": "Commercial Office Space - Hi-Tech Hub, Ra'anana",
        "text": (
            "Open-plan office space of 350 sqm in a modern hi-tech business park in Ra'anana. "
            "Features raised flooring, server room, conference rooms, and covered parking for 10 cars. "
            "Available for rent at 85 ILS/sqm/month. Suitable for tech companies up to 60 employees."
        ),
        "type": "office", "city": "Ra'anana", "rooms": 0, "price_ils": 0,
    },
    {
        "id": "listing_05",
        "title": "2-Bedroom Apartment in Haifa - Carmel Area",
        "text": (
            "Well-maintained 2-bedroom apartment on the Carmel mountain in Haifa. "
            "80 sqm, renovated bathroom, parquet floors, and a small balcony with garden view. "
            "Close to Haifa University. Asking price 1,450,000 ILS."
        ),
        "type": "apartment", "city": "Haifa", "rooms": 2, "price_ils": 1450000,
    },
    {
        "id": "listing_06",
        "title": "5-Room Penthouse - Ramat Gan Diamond Exchange",
        "text": (
            "Exclusive 5-room penthouse with 200 sqm terrace on the 25th floor in Ramat Gan. "
            "4 bedrooms, 3 bathrooms, chef's kitchen, and a private jacuzzi on the terrace. "
            "Panoramic city views. Asking 12,500,000 ILS."
        ),
        "type": "penthouse", "city": "Ramat Gan", "rooms": 5, "price_ils": 12500000,
    },
    {
        "id": "listing_07",
        "title": "Industrial Warehouse - Ashdod Port Zone",
        "text": (
            "800 sqm industrial warehouse near Ashdod port. 8-metre ceiling height, "
            "three loading docks, heavy electricity supply, and office annex. "
            "Zoned for light industry and logistics. Monthly rent 18,000 ILS."
        ),
        "type": "industrial", "city": "Ashdod", "rooms": 0, "price_ils": 0,
    },
    {
        "id": "listing_08",
        "title": "3-Bedroom Garden Apartment - Kfar Shmaryahu",
        "text": (
            "Ground-floor garden apartment with a 120 sqm private garden in Kfar Shmaryahu. "
            "3 bedrooms, 2 bathrooms, storage room, and two parking spots. Quiet street, "
            "walking distance to the beach. Asking 6,300,000 ILS."
        ),
        "type": "apartment", "city": "Kfar Shmaryahu", "rooms": 3, "price_ils": 6300000,
    },
    {
        "id": "listing_09",
        "title": "Retail Space - Ben Yehuda Street, Tel Aviv",
        "text": (
            "120 sqm retail unit on the famous Ben Yehuda pedestrian street in Tel Aviv. "
            "High foot traffic, glass storefront, back storage, and accessible entrance. "
            "Previous tenant was a café. Monthly rent 35,000 ILS."
        ),
        "type": "retail", "city": "Tel Aviv", "rooms": 0, "price_ils": 0,
    },
    {
        "id": "listing_10",
        "title": "4-Bedroom House - Moshav Beit Yanai Beachfront",
        "text": (
            "Single-family home on a 600 sqm plot directly on the beach at Moshav Beit Yanai. "
            "4 bedrooms, 2 bathrooms, open kitchen, large living room, and a sea-facing terrace. "
            "Asking 8,200,000 ILS. Requires light renovation."
        ),
        "type": "house", "city": "Beit Yanai", "rooms": 4, "price_ils": 8200000,
    },
    {
        "id": "listing_11",
        "title": "Luxury 3-Bedroom Apartment - Jaffa Old Port",
        "text": (
            "Restored heritage apartment in a 19th-century building in Old Jaffa. "
            "High ceilings, exposed stone walls, chef's kitchen, 2 bathrooms, and a private rooftop. "
            "Walking distance to galleries and restaurants. Asking 7,500,000 ILS."
        ),
        "type": "apartment", "city": "Jaffa", "rooms": 3, "price_ils": 7500000,
    },
    {
        "id": "listing_12",
        "title": "2-Bedroom New Build - Beer Sheva Northern Quarter",
        "text": (
            "Brand-new 2-bedroom apartment in a newly developed residential tower in Beer Sheva. "
            "72 sqm, energy-efficient, parking included, 15 years builder warranty. "
            "Asking 1,050,000 ILS. Ideal for first-time buyers."
        ),
        "type": "apartment", "city": "Beer Sheva", "rooms": 2, "price_ils": 1050000,
    },
    {
        "id": "listing_13",
        "title": "Villa with Pool - Modi'in",
        "text": (
            "Spacious 5-bedroom family villa in Modi'in with a heated swimming pool and landscaped garden. "
            "300 sqm built area on 700 sqm plot. Master suite, home cinema room, double garage. "
            "Asking 6,800,000 ILS."
        ),
        "type": "villa", "city": "Modi'in", "rooms": 5, "price_ils": 6800000,
    },
    {
        "id": "listing_14",
        "title": "Boutique Hotel Conversion Opportunity - Eilat",
        "text": (
            "Former guesthouse of 600 sqm in Eilat near the Red Sea marina. 12 rooms, "
            "reception area, rooftop terrace with sea view. Suitable for conversion to a boutique hotel "
            "or serviced apartments. Asking 9,000,000 ILS."
        ),
        "type": "commercial", "city": "Eilat", "rooms": 12, "price_ils": 9000000,
    },
    {
        "id": "listing_15",
        "title": "4-Room Apartment - Florentine, South Tel Aviv",
        "text": (
            "Renovated 4-room apartment in the trendy Florentine neighbourhood. "
            "90 sqm, exposed brick, modern kitchen, master bedroom with ensuite, large living area. "
            "Great for young professionals. Asking 3,200,000 ILS."
        ),
        "type": "apartment", "city": "Tel Aviv", "rooms": 4, "price_ils": 3200000,
    },
    {
        "id": "listing_16",
        "title": "3-Bedroom Apartment - Netanya Beachfront Tower",
        "text": (
            "High-floor apartment in a beachfront tower in Netanya with unobstructed sea views. "
            "3 bedrooms, 2 bathrooms, fully renovated kitchen, 2 parking spots. "
            "Asking 3,800,000 ILS."
        ),
        "type": "apartment", "city": "Netanya", "rooms": 3, "price_ils": 3800000,
    },
    {
        "id": "listing_17",
        "title": "Office Building - Petah Tikva Hi-Tech Park",
        "text": (
            "5-floor office building of 2,000 sqm in Petah Tikva. Each floor is 400 sqm open plan. "
            "Central AC, generator backup, 40-car parking lot. Suitable for corporate headquarters. "
            "Asking 28,000,000 ILS."
        ),
        "type": "office", "city": "Petah Tikva", "rooms": 0, "price_ils": 28000000,
    },
    {
        "id": "listing_18",
        "title": "1-Bedroom Apartment - Rothschild Boulevard, Tel Aviv",
        "text": (
            "Charming 1-bedroom apartment in a Bauhaus building on Rothschild Boulevard. "
            "55 sqm, high ceilings, original parquet floors, small balcony. "
            "Close to cafés and public transport. Asking 2,800,000 ILS."
        ),
        "type": "apartment", "city": "Tel Aviv", "rooms": 1, "price_ils": 2800000,
    },
    {
        "id": "listing_19",
        "title": "Agricultural Land - Jezreel Valley",
        "text": (
            "15-dunam agricultural plot in the Jezreel Valley with access road and water connection. "
            "Flat terrain suitable for crops or greenhouses. Zoned agricultural. "
            "Asking 1,800,000 ILS."
        ),
        "type": "land", "city": "Jezreel Valley", "rooms": 0, "price_ils": 1800000,
    },
    {
        "id": "listing_20",
        "title": "3-Bedroom Apartment - Rehovot Near Weizmann Institute",
        "text": (
            "Quiet 3-bedroom apartment near the Weizmann Institute in Rehovot. "
            "85 sqm, renovated bathrooms, parquet floors, 1 parking spot, storage unit. "
            "Suitable for academics and families. Asking 1,750,000 ILS."
        ),
        "type": "apartment", "city": "Rehovot", "rooms": 3, "price_ils": 1750000,
    },
    {
        "id": "listing_21",
        "title": "5-Room Duplex - Ramat HaSharon",
        "text": (
            "Two-level duplex apartment with a private garden in Ramat HaSharon. "
            "5 rooms, 3 bathrooms, American kitchen, and a large sunken living room. "
            "200 sqm total. Asking 7,200,000 ILS."
        ),
        "type": "apartment", "city": "Ramat HaSharon", "rooms": 5, "price_ils": 7200000,
    },
    {
        "id": "listing_22",
        "title": "Retail Strip Mall - Rishon LeZion",
        "text": (
            "Strip of 5 retail units totalling 600 sqm in a busy commercial district of Rishon LeZion. "
            "Four units currently tenanted with long leases. Estimated rental income 60,000 ILS/month. "
            "Asking 12,000,000 ILS."
        ),
        "type": "retail", "city": "Rishon LeZion", "rooms": 0, "price_ils": 12000000,
    },
    {
        "id": "listing_23",
        "title": "4-Bedroom House - Zichron Ya'akov",
        "text": (
            "Stone-built house in the historic town of Zichron Ya'akov. "
            "4 bedrooms, 2 bathrooms, wrap-around porch, wine cellar, and a garden with mature trees. "
            "Mountain and sea views. Asking 4,500,000 ILS."
        ),
        "type": "house", "city": "Zichron Ya'akov", "rooms": 4, "price_ils": 4500000,
    },
    {
        "id": "listing_24",
        "title": "3-Bedroom Apartment - Givat Shmuel",
        "text": (
            "Well-located 3-bedroom apartment in Givat Shmuel adjacent to Bar-Ilan University. "
            "82 sqm, renovated kitchen, 1 bathroom, 1 parking, elevator building. "
            "Asking 1,950,000 ILS."
        ),
        "type": "apartment", "city": "Givat Shmuel", "rooms": 3, "price_ils": 1950000,
    },
    {
        "id": "listing_25",
        "title": "Luxury Penthouse - Caesarea",
        "text": (
            "Stunning penthouse in a gated community in Caesarea with golf course views. "
            "5 bedrooms, 4 bathrooms, 300 sqm private terrace, infinity pool, smart-home system. "
            "Asking 18,000,000 ILS. Concierge service included."
        ),
        "type": "penthouse", "city": "Caesarea", "rooms": 5, "price_ils": 18000000,
    },
]


def seed():
    print(f"Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Delete existing collection and recreate
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")

    documents = []
    for listing in LISTINGS:
        doc = Document(
            page_content=listing["text"],
            metadata={
                "id": listing["id"],
                "title": listing["title"],
                "type": listing["type"],
                "city": listing["city"],
                "rooms": listing["rooms"],
            },
        )
        documents.append(doc)

    print(f"Embedding and storing {len(documents)} listings...")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
    )
    print(f"Done. Collection '{COLLECTION_NAME}' stored at: {CHROMA_PATH}")


if __name__ == "__main__":
    seed()
