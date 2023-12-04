import streamlit as st
from fruits_classifier import FruitClassifierModel

from utils import (show_details, get_tag, load_styles,
                   visualize_predictions_in_bars, show_final_prediction)
from utils import (display_properties, show_entry_msg, get_tag_and_image,
                   get_tag, get_image, show_final_prediction)

# --- GENERAL SETTINGS ---
st.session_state['cur_image_path'] = None
post_image = None

# --- PAGE SETTINGS ---
TITLE = 'Application of Convolutional Neural Network (CNN) on the Post-harvest Changes of Some Selected Climateric Fruits'

st.set_page_config(page_title='Fruit_Spoilage',
                   page_icon='fruit_icon.png',
                   layout='centered')
st.write(load_styles(), unsafe_allow_html=True)


# --- Loading Trained model ---
@st.cache_resource
def load_trained_model(model_dir='models'):
    model = FruitClassifierModel(model_dir_or_url=model_dir)
    return model


fruit_classifier_model = load_trained_model('models')


# --- MAIN PAGE
st.write(f"## {TITLE}")
st.write(show_details(), unsafe_allow_html=True)

st.write("#")
st.write(show_entry_msg(), unsafe_allow_html=True)

st.write("---")
choices = ('--- select Algorithm ---',
           'CNN')
algorithm_type = st.selectbox("Select Problem Type: ", choices)
st.write("---")


if algorithm_type.upper() == 'CNN':
    post_image = get_image(st)
    if post_image is not None:

        # TO-DO: prediction
        results = fruit_classifier_model.predict_pipeline(
            img_path=st.session_state['cur_image_path'])

        st.write('#')
        st.write('### Image Classification Results')
        st.write('---')
        st.write(results)
        visualize_predictions_in_bars(
            st, prediction_results=results['category'])
        visualize_predictions_in_bars(st, prediction_results=results['day'])

        # st.markdown(display_properties(
        #     results['properties']), unsafe_allow_html=True)

        cols = st.columns([2, 2, 2], gap='small')
        for i, (category, properties) in enumerate(results['properties'].items()):
            cols[i].markdown(f"""
                - **{category if len(category) <= 22 else category[:22] + '...'}**
                ---
                <div style="width:100%; padding:10px;">
            """, unsafe_allow_html=True)

            for key, val in properties.items():
                cols[i].markdown(f"""
                    <div style="padding:5px; background-color:#444; border-radius:5px; margin-bottom:10px;">
                        <div style="width:100%; padding:2px; color:#cccccc; ">
                            {key}
                        </div>
                        <div style="width:100%; padding:5px 10px; color:#ffffff; background-color:#322514;">
                            {val}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            cols[i].markdown(f"""
                </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
                <div style="color:#ffffff; background-color:#aaa; padding:10px; border-radius:10px;">
                    <h2 style="border-bottom:1px solid #555;">Final Prediction</h2>
                    <h4>{results['category']['prediction']} - {results['day']['prediction']}</h4>
                    <div style="color:#ddd; border-radius:5px; padding:10px; background-color:#333; 
                        display: inline-block;">
                        Confidence: {round(results['day']['confidence'])}%
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # # st.write(prediction_results=results['properties'])
        # st.write('---')

        # st.markdown(show_final_prediction(
        #     predicted=f"{results['category']['prediction']} - {results['day']['prediction']}",
        #     confidence=results['day']['confidence']), unsafe_allow_html=True)

    else:
        st.error("No image data received.")

elif algorithm_type.lower() == 'tag and image':
    data = get_tag_and_image(st)
    if data is not None:
        post_tag, post_image = data

        # TO-DO: prediction
        text_results, image_results = fruit_classifier_model.predict_hybrid(
            post=post_tag, img_path=st.session_state['cur_image_path'])

        avg_performance = fruit_classifier_model.get_avg_prediction_with_details(
            text_results, image_results)

        st.write('##')
        st.write('### Post Classification Results')
        st.write('---')

        col1, col2 = st.columns([2, 2], gap='medium')
        with col1:
            st.write('##### Text Classification Results (SVC)')
            st.write('---')
            visualize_predictions_in_bars(st, prediction_results=text_results)
            st.write('---')

        with col2:
            st.write('##### Image Classification Results (CNN)')
            st.write('---')
            visualize_predictions_in_bars(st, prediction_results=image_results)
            st.write('---')

        col1, col2 = st.columns([1, 1], gap='medium')
        with col1:
            st.write(text_results)

        with col2:
            st.write(image_results)

        st.write('---')

        st.write('####')
        st.write('## Hybrid Model Result (SVC + CNN)')
        st.write('---')
        st.write(avg_performance)
        visualize_predictions_in_bars(st, prediction_results=avg_performance)
        st.write('---')

        st.markdown(show_final_prediction(
            predicted=avg_performance['prediction'], confidence=avg_performance['confidence']), unsafe_allow_html=True)

    else:
        st.error("No image data received.")
