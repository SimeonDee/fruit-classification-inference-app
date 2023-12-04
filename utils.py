import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import path, listdir, remove


def load_styles():
    return """
        <style>
            .details-wrapper{
                margin: 20px 55px;
            }

            .about-app{
                font-size: 3rem;
                word-spacing: 2px;
                line-height: 2px;
                letter-spacing: 2px;
            }

            .footer{
                padding: 5px 10px;
                background-color: #444;
                color: #ccc;
                text-align: center;
            }

            .label{
                font-weight: bold;
            }

            td{
                padding: 5px 10px;
                font-size: large;

                border: 1px solid rgb(95, 81, 81);
                border-radius: 5px;
            }

            .input_div_wrapper, .input_div_wrapper2{
                padding:20px;
                min-height: 200px;
                border: 1px solid #555;
                border-radius: 10px;
            }

            .input_div_wrapper2{
                min-height: 320px;
            }

            h3{
                padding-bottom: 0px;
            }

            .final-prediction{
                font-size: xx-large;
                font-weight: bold;
                display: inline;
            }

            .final-prediction-confidence{
                background-color: #ccc;
                padding: 10px;
                border-radius: 10px;
                width: fit-content;
                color: #444;
                font-weight: bold;
            }

            .confidence-wrapper{
                border-top: 1px solid #ccc;
                padding-top: 5px;
            }

            .final-wrapper{
                padding: 20px 50px;
                border: 1px solid #888;
                border-radius: 15px;
                width: 100%;
                background-color: #444;
                color: #ccc;
            }
            
            .final-wrapper h4{
                color: #eee;
            }

            .entry-msg-wrapper{
                padding: 10px 20px;
                margin-top: 20px;
                margin-bottom: 0px;
                border: 1px solid #888;
                border-radius: 5px;
                width: 100%;
                background-color: #444;
                color: #ccc;
                text-align: center;
            }

            .section{
                display: flex;
                flex-direction: column;
                padding: 5px 10px;
                background-color: #322514;
            }
            .properties-wrapper{
                display: flex;
                flex-direction: row;
                background-color: #b20238;
                color: #eeeeee;
                /* border: 1px solid #bbbbbb; */
                border-radius: 10px;
                padding: 10px;
                width: 30%;
            }

            .property{
                padding: 5px 2px;
            }

            .property > .prop-key{
                color: #888888;
                font-size: small;
            }

        </style>
    """


def show_details():
    return """
        <div class="details-wrapper">
            <table>
                <tr>
                    <td class="label">Main Researcher:</td>
                    <td class="value">Ifeoluwa Pele (PhD)</td>
                </tr>
                <tr>
                    <td class="label">Co-Researcher:</td>
                    <td class="value">Adedoyin Simeon Adeyemi</td>
                </tr>
                <tr>
                    <td class="label">Institution:</td>
                    <td class="value">The Federal Polytrechnic, Offa, Kwara State, Nigeria</td>
                </tr>
                <tr>
                    <td class="label">Date:</td>
                    <td class="value">August, 2023</td>
                </tr>
                <tr>
                    <td class="label">Research Grant Body:</td>
                    <td class="value">TETFUND</td>
                </tr>
                <tr>
                    <td class="label">Research Category:</td>
                    <td class="value">IBR</td>
                </tr>
                <tr>
                    <td class="label">Sponsored By:</td>
                    <td class="value">TETFUND</td>
                </tr>
            </table>
        </div>
    """


def display_properties(properties):
    html = ''
    html = html + f"""
            <div class="section">
        """
    for category, prop in properties.items():
        html = html + f"""
                <div class="properties-wrapper">
                    <h4>{category}</h4>
                    <hr>
            """
        for k, v in prop.items():
            html = html + f"""
                    <div class="property">
                        <p class="prop-key">{k}</p>
                        <p class="prop-value">{v}</p>
                    </div>
                """
        html = html + f"""
                </div>
            """
    html = html + f"""
            </div>
        """

    return html


def show_entry_msg():
    return """
        <div class="entry-msg-wrapper">
            <p class="final-prediction"> Welcome to make your predictions </p>
        </div> 
    """


def show_final_prediction(predicted, confidence):
    return f"""
        <div class="final-wrapper">
            <h4 style="display: inline;">Final Prediction: </h4> 
            <div class="confidence-wrapper">
                <p class="final-prediction">{predicted}</p>
                <div class="final-prediction-confidence"><em>{int(np.round(confidence, 0))}% confident</em></div>
            </div>
        </div>
    """


def visualize_predictions_in_bars(st, prediction_results):
    # try:
    # Plotting first five predictions
    x = prediction_results['probability'] if len(
        prediction_results['probability']) > 1 else prediction_results['probability'][0]
    y = prediction_results['classes']

    series = pd.Series(x, index=y)
    series.sort_values(ascending=False, inplace=True)

    colors = [
        '#191970', '#32CD32', '#00FF00', '#7CFC00', '#7FFF00', '#ADFF2F', '#8B0000',
        '#B22222', '#FF0000', '#DC143C', '#CD5C5C', '#F08080', '#E9967A',
        '#FA8072', '#FFA07A', '#FF7F50', '#FF8C00', '#FFA500', '#FFD700', '#FFFF00'
    ]
    # colors[np.argmax(x)] = 'blue'
    # colors[np.argmin(x)] = 'brown'

    fig = plt.figure(figsize=(3, 1))
    plt.bar(x=series.index, height=series.values, color=colors[:len(y)])
    # plt.title(
    #     f'Prediction: {y[np.argmax(x)]} \n(Confidence: {np.round(max(x) * 100, 2)}%)')
    plt.title(
        f'Prediction: {prediction_results["prediction"]} \n(confidence: {prediction_results["confidence"]}%)',
        fontsize=8)

    plt.xticks(rotation=45, fontsize=6)
    plt.yticks(fontsize=6)
    st.write(fig)


def save_image(st, filename, data):

    if len(listdir('uploads')) > 0:  # if any file
        for file in listdir('uploads'):
            # delete any exising file(s) to free storage
            remove(path.join('uploads', file))

    file_path = path.join('uploads', filename.replace(' ', ''))

    try:
        with open(file_path, 'wb') as f:
            f.write(data)

        if path.exists(file_path):  # if saved
            st.session_state['cur_image_path'] = file_path
        else:
            st.error("File not saved")
    except:
        st.error("Error: Error saving file")


# --- TEXT ONLY ---
def get_tag(st):
    st.write('####')
    st.markdown('### Kindly fill the form below')
    st.write('---')
    tag = st.text_area("Enter your post text:",
                       placeholder='Enter post text here', height=200)

    st.write('---')
    submit_text_button = st.button("Classify Text")
    st.write('---')

    if submit_text_button:
        if tag:
            st.write('####')
            st.write("### Input Data Received")
            st.write('---')

            input_div_wrapper = f"""
                <div class="input_div_wrapper">
                    {tag}
                </div>
            """
            st.write(input_div_wrapper, unsafe_allow_html=True)
            st.write('---')

            return tag

        else:
            st.error("Text post field cannot be empty")

    return tag


# --- IMAGE ONLY ---
def get_image(st):
    # st.write('####')
    st.markdown('### Kindly Upload the Image below')
    st.write('---')

    uploaded_image = st.file_uploader(
        "Upload the image file", accept_multiple_files=False, type=['jpg', 'png', 'jpeg'])

    if uploaded_image is not None:
        image_bytes = uploaded_image.getvalue()
        st.image(image_bytes, caption='Loaded Image', width=200)

        st.write('---')
        submit_image_button = st.button("Classify Image")
        st.write('---')

        if submit_image_button:
            if uploaded_image:
                st.write('####')
                st.write("### Input Image Received")
                st.write('---')
                st.image(image_bytes, caption='Loaded Image', width=200)
                st.write(f"""
                    File Name: <strong><em>{uploaded_image.name}</em></strong><br>
                    File Type: <strong><em>{uploaded_image.type}</em></strong><br>
                    File Size: <strong><em>{np.round(uploaded_image.size / 1024, 1)} kb</em></strong>
                """, unsafe_allow_html=True)
                st.write('---')

                save_image(st, filename=uploaded_image.name, data=image_bytes)

                return uploaded_image

            else:
                st.error("Text post field cannot be empty")
                return None
    else:
        st.error("No image uploaded yet")
        return None


# --- TEXT AND IMAGE ---
def get_tag_and_image(st):
    st.write('####')
    st.markdown('### Kindly fill the form below')
    st.write('---')

    tag = st.text_area("Enter your post text:",
                       placeholder='Enter post text here', height=200)

    uploaded_image = st.file_uploader(
        "Upload the image file", accept_multiple_files=False, type=['jpg', 'png', 'jpeg'])

    if uploaded_image is not None and tag is not None:
        st.write('---')
        submit_post_button = st.button("Classify Post")
        st.write('---')

        image_bytes = uploaded_image.getvalue()
        st.image(image_bytes, caption='Loaded Image', width=200)

        if submit_post_button:
            if uploaded_image:
                st.write('####')
                st.write("### Input Post Data Received")
                st.write('---')

                col1, col2 = st.columns([2, 5], gap='medium')

                with col1:
                    st.image(image_bytes, caption='Loaded Image', width=200)
                    st.write(f"""
                        File Name: <strong><em>{uploaded_image.name}</em></strong><br>
                        File Type: <strong><em>{uploaded_image.type}</em></strong><br>
                        File Size: <strong><em>{np.round(uploaded_image.size / 1024, 1)} kb</em></strong>
                    """, unsafe_allow_html=True)

                    save_image(st, filename=uploaded_image.name,
                               data=image_bytes)

                with col2:
                    input_div_wrapper = f"""
                        <div class="input_div_wrapper2">
                            {tag}
                        </div>
                    """
                    st.write(input_div_wrapper, unsafe_allow_html=True)
                st.write('---')

                return tag, uploaded_image

            else:
                st.error("Text post field cannot be empty")
                return None
    else:
        st.error("No image uploaded yet")
        return None
