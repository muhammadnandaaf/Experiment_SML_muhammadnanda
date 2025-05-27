import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import joblib
import os

def preprocessing_pipeline(csv_path):
    df = pd.read_csv(csv_path, encoding='latin1')

    # 1. Drop Fitur yang tidak digunakan
    df = df.drop(['ApplicationDate'], axis=1)

    # 2. Splitting data
    train_df, test_df = train_test_split(df, test_size=0.05, random_state=42, shuffle=True)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # 3. Oversampling
    df_majority = train_df[(train_df.LoanApproved == 0)]
    df_minority = train_df[(train_df.LoanApproved == 1)]

    df_minority_oversampled = resample(df_minority, n_samples=14484, random_state=42)

    oversampled_train_df = pd.concat([df_majority, df_minority_oversampled]).reset_index(drop=True)
    oversampled_train_df = shuffle(oversampled_train_df, random_state=42)
    oversampled_train_df.reset_index(drop=True, inplace=True)

    # 4. Fitur Target
    X_train = oversampled_train_df.drop(columns="LoanApproved", axis=1)
    y_train = oversampled_train_df["LoanApproved"]
 
    X_test = test_df.drop(columns="LoanApproved", axis=1)
    y_test = test_df["LoanApproved"]

    # 5. Numerikal & Kategorikal Column
    numerical_columns = X_train.select_dtypes(include=["int64", "float64"]).columns.drop('LoanApproved', errors='ignore')
    categorical_columns = X_train.select_dtypes(include=["object"]).columns.drop('LoanApproved', errors='ignore')

    # Ubah ke dalam dataframe
    numerical_df = pd.DataFrame(df[numerical_columns])
    categorical_df = pd.DataFrame(df[categorical_columns])

    # 6. Scaling Numerikal
    def scaling(features, df, df_test=None):
        os.makedirs("model/scaling", exist_ok=True)
        
        if df_test is not None:
            df = df.copy()
            df_test = df_test.copy()
            for feature in features:
                scaler = MinMaxScaler()
                X = np.asanyarray(df[feature])
                X = X.reshape(-1,1)
                scaler.fit(X)
                df["{}".format(feature)] = scaler.transform(X)
                joblib.dump(scaler, "model/scaling/scaler_{}.joblib".format(feature))
                
                X_test = np.asanyarray(df_test[feature])
                X_test = X_test.reshape(-1,1)
                df_test["{}".format(feature)] = scaler.transform(X_test)
                print(f"Scaling feature: {feature}")
            return df, df_test
        else:
            df = df.copy()
            for feature in features:
                scaler = MinMaxScaler()
                X = np.asanyarray(df[feature])
                X = X.reshape(-1,1)
                scaler.fit(X)
                df["{}".format(feature)] = scaler.transform(X)
                joblib.dump(scaler, "model/scaling/scaler_{}.joblib".format(feature))
            return df
    X_train, X_test = scaling(numerical_columns, X_train, X_test)
    
    # 7. Encoding Kategorikal
    def encoding(features, df, df_test=None):
        os.makedirs("model/encoding", exist_ok=True)

        if df_test is not None:
            df = df.copy()
            df_test = df_test.copy()
            for feature in features:
                encoder = LabelEncoder()
                
                # Gabungkan data training dan testing untuk menghindari unseen labels
                combined_data = pd.concat([df[feature], df_test[feature]], axis=0)
                encoder.fit(combined_data)

                df[feature] = encoder.transform(df[feature])
                df_test[feature] = encoder.transform(df_test[feature])
                
                joblib.dump(encoder, f"model/encoding/encoder_{feature}.joblib")
                print(f"Encoding feature: {feature}")
            return df, df_test
        else:
            df = df.copy()
            for feature in features:
                encoder = LabelEncoder()
                encoder.fit(df[feature])
                df[feature] = encoder.transform(df[feature])
                joblib.dump(encoder, f"model/encoding/encoder_{feature}.joblib")
            return df
    X_train, X_test = encoding(categorical_columns, X_train, X_test)

    # 8. Reduksi PCA
        # Demografi
    pca_numericalColumns_1 = [
        "Age",
        "Experience",
        "CreditScore"
    ]
        # Saldo Akun
    pca_numericalColumns_2 = [
        "SavingsAccountBalance",
        "CheckingAccountBalance",
        "TotalAssets"
    ]
        # Beban Utang
    pca_numericalColumns_3 = [
        "InterestRate",
        "DebtToIncomeRatio",
        "TotalDebtToIncomeRatio",
        "RiskScore"
    ]
        # Informasi Peminjaman
    pca_numericalColumns_4 = [
        "LoanAmount",
        "MonthlyLoanPayment"
    ]
    train_pca_df = X_train.copy().reset_index(drop=True)
    test_pca_df = X_test.copy().reset_index(drop=True)

    # 9. Fungsi PCA
    def apply_pca_and_transform(pca_features, train_df, test_df, pca_name):
        """ 
        Melakukan PCA dengan jumlah komponen efisien (min. 95% variansi), 
        menyimpan model, dan menambahkan principal components ke DataFrame.
        """
        # Inisialisasi PCA tanpa mengatur n_components
        pca = PCA(random_state=123)
        pca.fit(train_df[pca_features])

        # Hitung variansi kumulatif
        var_exp = pca.explained_variance_ratio_.round(3)
        cum_var_exp = np.cumsum(var_exp)
        
        # Tentukan jumlah komponen efisien (>=95% variansi kumulatif)
        n_components = np.argmax(cum_var_exp >= 0.95) + 1
        print(f"🔍 Optimal number of components for {pca_name}: {n_components} (cumulative variance ≥ 95%)")

        # Ulang PCA dengan n_components optimal
        pca = PCA(n_components=n_components, random_state=123)
        pca.fit(train_df[pca_features])
        
        # Simpan model PCA
        joblib.dump(pca, f"model/pca_{pca_name}.joblib")
        print(f"📌 PCA Model for '{pca_name}' saved as 'model/pca_{pca_name}.joblib'.")
        
        # Transform data
        princ_comp_train = pca.transform(train_df[pca_features])
        princ_comp_test = pca.transform(test_df[pca_features])

        # Membuat nama kolom baru
        pca_columns = [f"{pca_name}_{i+1}" for i in range(n_components)]
        
        # Tambahkan Principal Components ke DataFrame
        train_df[pca_columns] = pd.DataFrame(princ_comp_train, columns=pca_columns, index=train_df.index)
        test_df[pca_columns] = pd.DataFrame(princ_comp_test, columns=pca_columns, index=test_df.index)
        
        # Drop kolom asli yang digunakan di PCA
        train_df.drop(columns=pca_features, inplace=True)
        test_df.drop(columns=pca_features, inplace=True)
        print(f"\n📌 Features for '{pca_name}' successfully replaced with {n_components} Principal Components.\n")

        return train_df, test_df

    # 10. Eksekusi PCA
    train_pca_df, test_pca_df = apply_pca_and_transform(
    pca_features=pca_numericalColumns_1, 
    train_df=train_pca_df, 
    test_df=test_pca_df,
    pca_name="pc1"
    )
    train_pca_df, test_pca_df = apply_pca_and_transform(
        pca_features=pca_numericalColumns_2, 
        train_df=train_pca_df, 
        test_df=test_pca_df,
        pca_name="pc2"
    )
    train_pca_df, test_pca_df = apply_pca_and_transform(
        pca_features=pca_numericalColumns_3, 
        train_df=train_pca_df, 
        test_df=test_pca_df,
        pca_name="pc3"
    )
    train_pca_df, test_pca_df = apply_pca_and_transform(
        pca_features=pca_numericalColumns_4, 
        train_df=train_pca_df, 
        test_df=test_pca_df,
        pca_name="pc4"
    )

    # 11. Simpan Hasil
    # Gabungkan fitur dengan target sebelum menyimpan (opsional tapi umum)
    y_train = oversampled_train_df['LoanApproved']
    y_test = test_df['LoanApproved']
    train_final = pd.concat([train_pca_df, y_train.reset_index(drop=True)], axis=1)
    test_final = pd.concat([test_pca_df, y_test.reset_index(drop=True)], axis=1)

    train_final.to_csv('preprocessing/train_pca.csv', index=False)
    test_final.to_csv('preprocessing/test_pca.csv', index=False)

    return train_final, test_final

file_path = f'datasets/Loan.csv'
train_final, test_final = preprocessing_pipeline(file_path)
train_final.to_csv("train_pca.csv", index=False)
test_final.to_csv("test_pca.csv", index=False)