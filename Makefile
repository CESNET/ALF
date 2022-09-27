PYTEST_BIN=pytest
TEST_OPTIONS="-s"
PIP=pip
REQUIREMENTS=requirements.txt
REQUIREMENTS_NEMEA=requirements.nemea.txt
REQUIREMENTS_DEV=requirements.dev.txt

LINTER_BIN=flake8

init:
	$(PIP) install -r $(REQUIREMENTS) -r $(REQUIREMENTS_NEMEA) -r $(REQUIREMENTS_DEV)

test:
	mkdir -p /tmp/alf
	rm -f /tmp/alf/*
	$(PYTEST_BIN) $(TEST_OPTIONS) tests
	rm -f /tmp/alf/*

report:
	mkdir -p /tmp/alf
	rm -f /tmp/alf/*
	$(PYTEST_BIN) --html=report.html --self-contained-html tests
	mv report.html docs/_build/html/report.html
	rm -f /tmp/alf/*

lint:
	@echo "Linting:"
	$(LINTER_BIN) --count alf

send:
	/usr/bin/nemea/traffic_repeater -i "f:example.trapcap,u:alf_predictor_socket"

receive:
	/usr/bin/nemea/logger -i "u:sink" -t




