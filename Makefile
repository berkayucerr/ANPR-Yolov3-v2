all: run
cls:
	rm -rf __pycache__
	rm -rf utils/__pycache__
	rm test_data/.DS_Store
	@echo "--------Cache files are removed--------"
run:
	@echo "Project is running..."
	python image.py
k:
	killall python
	@echo "--------Python processes are killed--------"