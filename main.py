import streamlit as st
from streamlit_drawable_canvas import st_canvas
from model_pred import Letter_model
from utils import save_image


@st.cache_resource
def load_and_start_model(path):
    md_letter = Letter_model(path=path)
    md_letter.init_model()  # Carrega o modelo na primeira vez
    return md_letter


if __name__ == "__main__":
    st.set_page_config(
        page_title="Handwritten Letter Prediction", layout="wide")

    st.write("""
        # ✍️ Handwritten letter and numbers predicition

        This project was coded to predict letters and numbers that were handwritten. Draw some letters or numbers below, and it will show you which letter it is.

        - Draw a :red[letter] or a :blue[number] in canvas
        - :red[Press] the button for predict
        - Wait for :blue[prediction]
        - :red[Clear] canva and draw another letter or number
        - Have :red[fun]! 😁

        ---

        **Hints for better predictions:**
        
        - Try to make a drawing that occupies the maximum area of the square.
        - The model was trained with uppercase cursive alphabetic letters, so try to draw the same way.
        - Because they are cursive, the model may show incorrect results. Thats why i put 3 probabily results

    """)

    lf_column, rg_column = st.columns(2)

    with lf_column:

        st.header(":blue[Canva Draw]")

        canva = st_canvas(
            background_color="rgba(255, 255, 255, 1)", stroke_width=20, width=300, height=300)
        pred_button = st.button('Predict what is the letter/number is',
                                use_container_width=True, icon="🤔")

    with rg_column:

        st.header(':red[Prediction]')

        st.subheader('🤖 My prediction is that it is the letter/number:')

        if pred_button:

            # data = canva.image_data
            # data = Image.fromarray(data)
            # data = data.resize((1200, 900))
            # data.save('temp_img.png')
            save_image(canva.image_data)

            predicition, probabilities = load_and_start_model(path='./saved_model/checkpoint_model.h5').predict(
                img_path='temp_img.png')

            html_content = ''.join([
                f'''
                <div style="display:flex;flex-direction:column;justify-content:center;align-items:center;width:100%; height:250px">
                    <p style="font-size:5rem">{predicition[i]}</p>
                    <p >{probabilities[i] *100:0.2f}%</p>
                </div>
                ''' for i in range(len(predicition))
            ])
            st.markdown(f'''
                <div style='display:flex;'>
                    {html_content}
                </div>
            ''', unsafe_allow_html=True)
