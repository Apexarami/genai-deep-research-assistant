install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload

ui:
	streamlit run frontend/streamlit_app.py

test:
	pytest
