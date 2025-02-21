from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Carica il modello di Regressione Logistica e lo StandardScaler all'avvio dell'app
modello_path = 'models/modello_regress_logistic_recall-80_F1-45.pkl' # Assicurati che il percorso sia corretto
scaler_path = 'models/scaler_reg_logistic_rec-80_F1-45.pkl' # Assicurati che il percorso sia corretto

try:
    with open(modello_path, 'rb') as modello_file:
        model = pickle.load(modello_file)
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Modello e Scaler caricati con successo!") # Messaggio di conferma nel terminale
except Exception as e:
    print(f"Errore nel caricamento del modello o dello scaler: {e}")
    model = None
    scaler = None

@app.route('/', methods=['GET', 'POST']) # Permetti sia GET che POST
def index():
    prediction = None # Inizializza la variabile prediction a None

    if request.method == 'POST': # Se la richiesta è di tipo POST (form inviato)
        try:
            # Ottieni i dati dal form HTML
            age = float(request.form['age'])
            sex = int(request.form['sex']) # Assicurati di gestire correttamente i tipi di input nel form
            cholesterol = float(request.form['cholesterol'])
            hdl = float(request.form['hdl'])
            ldl = float(request.form['ldl'])
            systolic_bp = float(request.form['systolic_bp'])
            diastolic_bp = float(request.form['diastolic_bp'])
            smoking = int(request.form['smoking']) # Assicurati di gestire correttamente i tipi di input nel form
            diabetes = int(request.form['diabetes']) # Assicurati di gestire correttamente i tipi di input nel form

            # Crea un DataFrame pandas con i dati di input, rispettando l'ordine delle features usato per l'addestramento
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [sex],
                'total_cholesterol': [cholesterol],
                'ldl': [ldl],
                'hdl': [hdl],                
                'systolic_bp': [systolic_bp],
                'diastolic_bp': [diastolic_bp],
                'smoking': [smoking],
                'diabetes': [diabetes]
            })

            # Standardizza le features numeriche usando lo scaler caricato
            colonne_numeriche_da_scalare = ['age', 'total_cholesterol', 'ldl', 'hdl', 'systolic_bp', 'diastolic_bp']
            input_data[colonne_numeriche_da_scalare] = scaler.transform(input_data[colonne_numeriche_da_scalare])

            # Esegui la predizione con il modello caricato
            probability = model.predict_proba(input_data)[:, 1][0] # Ottieni la probabilità della classe positiva
            

            # Determina il testo della predizione da mostrare all'utente 
            if probability <= 0.28:
                prediction_text = f"Rischio Cardiaco: Basso (Probabilità: {probability*100:.2f}%) - Il rischio stimato è basso ma mantieni uno stile di vita sano."
            elif probability <= 0.35:
                prediction_text = f"Rischio Cardiaco: Abbastanza Basso (Probabilità: {probability*100:.2f}%) - Il rischio stimato è abbastanza basso ma mantieni uno stile di vita sano e consulta regolarmente il tuo medico."
            elif probability <= 0.39:
                prediction_text = f"Rischio Cardiaco: Moderatamente Alto (Probabilità: {probability*100:.2f}%) - Il rischio stimato è moderatamente alto, sarebbe opportuno parlarne con il proprio medico."
            elif probability <= 0.5:
                prediction_text = f"Rischio Cardiaco: Abbastanza Alto (Probabilità: {probability*100:.2f}%) - Il rischio stimato è abbastanza alto, si raccomanda di parlarne con il proprio medico per eventuali accertamenti."
            else:
                prediction_text = f"Rischio Cardiaco: Alto (Probabilità: {probability*100:.2f}%) - Il rischio stimato è alto, si raccomanda di consultare il proprio medico per ulteriori accertamenti."

            prediction = prediction_text # Assegna il testo della predizione alla variabile prediction

        except Exception as e:
            prediction = f"Errore durante la predizione: {e}" # Gestisci eventuali errori durante l'elaborazione del form

    return render_template('index.html', prediction=prediction) # Passa la variabile prediction al template


if __name__ == '__main__':
    app.run(debug=True)