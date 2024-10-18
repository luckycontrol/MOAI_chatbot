import streamlit as st
import re

def print_message():    
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        for role, message in st.session_state["messages"]:
            st.chat_message(role).write(message)

def process_text(text):
    # Find all brackets in the text
    brackets = re.findall(r'\[.*?\]', text)
    
    result = []
    
    if brackets:
        # If brackets are found, process them
        for bracket in brackets:
            # Split the text by the current bracket
            parts = text.split(bracket, 1)
            
            # Add the text before the bracket (if any) to the result
            if parts[0].strip():
                result.append(f"문장: {parts[0].strip()}")
            
            # Add the bracket as an image
            result.append(f"이미지: {bracket}")
            
            # Update the text to be the part after the bracket
            text = parts[1] if len(parts) > 1 else ""
    
    # Add any remaining text as a sentence
    if text.strip():
        result.append(f"문장: {text.strip()}")
    
    # Join the results with newlines
    return result