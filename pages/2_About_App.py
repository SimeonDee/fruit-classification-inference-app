import streamlit as st


def load_styles():
    return """
        <style>
            .details-wrapper{
                margin: 20px 55px;
            }

            .about-app{
                margin-top: 25px;
                padding: 20px;
                padding-left: 0;
                font-size: 3rem;
                word-spacing: 2px;
                letter-spacing: 2px;
            }

            .footer{
                margin-top: 30px;
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


# --- Loading css styles ---
TITLE = 'Application of Convolutional Neural Network (CNN) on the Post-harvest Changes of Some Selected Climateric Fruits'

st.write(load_styles(), unsafe_allow_html=True)
st.write(f'# {TITLE}')
st.write(show_details(), unsafe_allow_html=True)

st.write("""
    #
    <h3> About App </h3>
    <div class="about-app">
         <p> This app classifies fruit image into its fruit type, estimated ripening day and its Physico-chemical properties, vitamins and Proximate Analysis using CNN algorithm.</p>
    </div>
         
    <div class="footer">
        <p> &copy; August 2023 </p>
    </div>
""", unsafe_allow_html=True)
