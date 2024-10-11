import streamlit as st

def main():
    st.title("Python UI with Streamlit")
    st.write("Welcome to the Streamlit application!")
    
    name = st.text_input("Enter your name:")
    if st.button("Submit"):
        st.write(f"Hello, {name}!")

if __name__ == "__main__":
    main()
