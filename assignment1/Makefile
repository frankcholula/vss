.PHONY: run
run:
	streamlit run visual_search.py --server.runOnSave true

.PHONY: test
test:
	pytest -s

.PHONY: clean
clean:
	rm -rf __pycache__ .pytest_cache
	rm -rf MSRC_ObjCategImageDatabase_v2_local/GroundTruth/labels.json
