import numpy as np

class LinearRegression:
    def __init__(self,learning_rate=0.01,n_iterations=1000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None

    def fit(self,X, y):
        """
        X shape: (n_samples, n_features) -> Contoh: (100 rumah, 2 fitur)
        y shape: (n_samples, 1)          -> Contoh: (100 harga rumah, 1 target)
        """
        # 1. Baca Dimensi Data
        # n_samples -> Akan dipakai nanti sbg pembagi rata-rata error
        # n_features -> Dipakai sbg baris di matriks Weights
        n_samples, n_features = X.shape
        
        # 2. Baca Dimensi Target (y) untuk menghindari hard-code
        # Jika y adalah (100, 1) -> n_targets = 1
        # Jika y adalah (100, 2) -> n_targets = 2 (Multi-Output)
        n_targets = y.shape[1]
        
        # 3. Lahirkan Weights dan Bias berdasarkan jumlah fitur
        # Jika X punya 2 fitur, maka weights jadi (2, 1) berisi angka 0
        self.weights = np.zeros((n_features, n_targets))
        self.bias = np.zeros((1, n_targets))
        
        # --- Sampai sini, kontrak dimensi sudah aman ---
        # --- FASE 2: TRAINING LOOP (GRADIENT DESCENT) ---
        for _ in range(self.n_iters):
            # Step 1: Forward Pass (Coba tebak pakai bobot saat ini)
            y_pred = self.forward(X)
            
            # Step 2: Hitung Error (Seberapa jauh melesetnya?)
            error = y_pred - y
            
            # Step 3: Hitung Gradient (Kalkulus Dasar: Cari arah perbaikan)
            # Di sinilah Transpose (.T) dan n_samples terpakai!
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error, axis=0)
            
            # Step 4: Update Parameter (Lakukan perbaikan)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
    def forward(self, X):
        """
        Input X: (n_samples, n_features)
        Output: (n_samples, 1)
        """
        # PILAR 6: Dot Product + Bias
        # Rumus: y_hat = X @ W + b
        if self.weights is None or self.bias is None:
            raise RuntimeError(
                "Model is not trained yet"
                )
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        """
        Alias untuk forward pass. 
        Digunakan setelah model dilatih (fit).
        """
        return self.forward(X)