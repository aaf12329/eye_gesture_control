const int A1_PIN = 4;
const int B1_PIN = 6;
const int P1_PIN = 9;

const int A2_PIN = 5;
const int B2_PIN = 7;
const int P2_PIN = 10;

void stopAll() {
  analogWrite(P1_PIN, 0);
  analogWrite(P2_PIN, 0);

  digitalWrite(A1_PIN, LOW);
  digitalWrite(B1_PIN, LOW);
  digitalWrite(A2_PIN, LOW);
  digitalWrite(B2_PIN, LOW);
}

void setup() {
  Serial.begin(9600);

  pinMode(A1_PIN, OUTPUT);
  pinMode(B1_PIN, OUTPUT);
  pinMode(P1_PIN, OUTPUT);

  pinMode(A2_PIN, OUTPUT);
  pinMode(B2_PIN, OUTPUT);
  pinMode(P2_PIN, OUTPUT);

  stopAll();

  Serial.println("Motor test ready");
  Serial.println("1=M1 forward, 2=M2 forward");
  Serial.println("3=M1 reverse, 4=M2 reverse");
  Serial.println("F=both forward, S=stop");
}

void loop() {
  if (!Serial.available()) {
    return;
  }

  char cmd = Serial.read();

  if (cmd == '1') {
    stopAll();
    digitalWrite(A1_PIN, LOW);
    digitalWrite(B1_PIN, HIGH);
    analogWrite(P1_PIN, 40);
    delay(1000);
    stopAll();
  }

  else if (cmd == '2') {
    stopAll();
    digitalWrite(A2_PIN, LOW);
    digitalWrite(B2_PIN, HIGH);
    analogWrite(P2_PIN, 40);
    delay(1000);
    stopAll();
  }

  else if (cmd == '3') {
    stopAll();
    digitalWrite(A1_PIN, HIGH);
    digitalWrite(B1_PIN, LOW);
    analogWrite(P1_PIN, 40);
    delay(1000);
    stopAll();
  }

  else if (cmd == '4') {
    stopAll();
    digitalWrite(A2_PIN, HIGH);
    digitalWrite(B2_PIN, LOW);
    analogWrite(P2_PIN, 40);
    delay(1000);
    stopAll();
  }

  else if (cmd == 'F') {
    stopAll();

    digitalWrite(A1_PIN, LOW);
    digitalWrite(B1_PIN, HIGH);
    digitalWrite(A2_PIN, LOW);
    digitalWrite(B2_PIN, HIGH);

    analogWrite(P1_PIN, 255);
    analogWrite(P2_PIN, 255);

    delay(2000);
    stopAll();
  }

  else if (cmd == 'S') {
    stopAll();
  }
}