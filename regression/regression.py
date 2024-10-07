import streamlit as st
import sklearn
import joblib, os
import numpy as np


st.set_page_config(
    page_title="Regression - Streamlit App",
    page_icon="❤️",
)

def load_prediction_model (model_file) :
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

def main() :
    """Regresi Linier Sederhana"""
    st.title('Penentuan Gaji Karyawan')

    html_templ = """<div style="background-color:#0047AB;padding:10px;">
                    <h3 style="color:white">Penentuan Gaji Karyawan Menggunakan Regresi Linier</h3>
                    </div>"""
    
    st.markdown(html_templ, unsafe_allow_html=True)
    activity = ["Penentuan Gaji Karyawan", "Apa itu Regresi?"]
    choice = st.sidebar.selectbox("Main", activity)

    if choice == 'Penentuan Gaji Karyawan':
        st.markdown('---')
        st.subheader('Penentuan Gaji Karyawan')
        experience = st.slider('Berapa tahun pengalaman kerjanya?', 0,20)

        #st.write(type(experience))
        if st.button('Proses'):
            regressor = load_prediction_model('linear_regression_salary.pkl')
            experience_reshaped = np.array(experience).reshape(-1, 1)
            predicted_salary = regressor.predict(experience_reshaped)

            st.info('Gaji untuk karyawan dengan pengalaman kerja {} tahun: {}'.format(experience,predicted_salary[0][0].round(2)))
    
    else :
        st.markdown('---')
        st.subheader('Apa itu Regresi?')
        st.write('')
        st.info('Regresi linear adalah teknik analisis data yang memprediksi nilai data yang tidak diketahui dengan '
         'menggunakan nilai data lain yang terkait dan diketahui. Secara matematis memodelkan variabel yang tidak '
         'diketahui atau tergantung dan variabel yang dikenal atau independen sebagai persamaan linier.')


if __name__ == '__main__':
    main()