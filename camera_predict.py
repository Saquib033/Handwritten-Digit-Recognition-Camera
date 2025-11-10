import cv2
import numpy as np
import tensorflow as tf

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("mnist_digit_model.h5")

# ---------------- LOAD & EVALUATE TEST SET ----------------
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()

x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
test_acc_percent = test_acc * 100

print(f"\n Test Loss     : {test_loss:.4f}")
print(f" Test Accuracy : {test_acc_percent:.2f}%\n")

# ---------------- START CAMERA ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Bounding Box for digit input
    cv2.rectangle(frame, (100,100), (300,300), (255,0,0), 2)
    digit = frame_gray[100:300, 100:300]

    # Pre-process for Model Prediction
    resized = cv2.resize(digit, (28,28))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)

    prediction = model.predict(reshaped, verbose=0)
    digit_class = np.argmax(prediction)

    # Display Prediction
    cv2.putText(frame, f"Prediction: {digit_class}", (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

    # Display Test Loss and Accuracy on Screen
    cv2.putText(frame, f"Test Loss: {test_loss:.4f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame, f"Test Accuracy: {test_acc_percent:.2f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("Handwritten Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
