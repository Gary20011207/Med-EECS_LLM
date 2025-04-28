#!/bin/bash
ollama serve &
sleep 2
ollama run deepseek-r1:8b
