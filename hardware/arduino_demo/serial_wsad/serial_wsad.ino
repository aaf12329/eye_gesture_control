// Open this sketch separately from ../ardino_demo.ino.
// Serial: 9600 baud. W/S = forward/reverse, A/D = turn, X/space = stop.
// Commands stay active until the next command. CR/LF are ignored.
// M1 is assumed to be the left motor; M2 is the right motor.

const int LEFT_A_PIN = 4;
const int LEFT_B_PIN = 6;
const int LEFT_PWM_PIN = 9;
const int RIGHT_A_PIN = 5;
const int RIGHT_B_PIN = 7;
const int RIGHT_PWM_PIN = 10;

// PWM range: 0..255. Keep the speed from the original sketch.
const int MOTOR_PWM = 40;
// Set the corresponding flag to true if a wheel runs backwards.
const bool LEFT_REVERSED = false;
const bool RIGHT_REVERSED = false;

void stopAll() {
  analogWrite(LEFT_PWM_PIN, 0);
  analogWrite(RIGHT_PWM_PIN, 0);
  digitalWrite(LEFT_A_PIN, LOW);
  digitalWrite(LEFT_B_PIN, LOW);
  digitalWrite(RIGHT_A_PIN, LOW);
  digitalWrite(RIGHT_B_PIN, LOW);
}

// direction: 1 = forward, -1 = reverse, 0 = stopped.
void setMotor(int pinA, int pinB, int pwmPin, int direction, bool reversed) {
  if (reversed) {
    direction = -direction;
  }
  // The original sketch defines LOW/HIGH as forward.
  digitalWrite(pinA, direction < 0 ? HIGH : LOW);
  digitalWrite(pinB, direction > 0 ? HIGH : LOW);
  analogWrite(pwmPin, direction == 0 ? 0 : MOTOR_PWM);
}

void drive(int leftDirection, int rightDirection) {
  // Disable PWM before changing direction.
  stopAll();
  delay(2);
  setMotor(LEFT_A_PIN, LEFT_B_PIN, LEFT_PWM_PIN, leftDirection, LEFT_REVERSED);
  setMotor(RIGHT_A_PIN, RIGHT_B_PIN, RIGHT_PWM_PIN, rightDirection, RIGHT_REVERSED);
}

void setup() {
  pinMode(LEFT_A_PIN, OUTPUT);
  pinMode(LEFT_B_PIN, OUTPUT);
  pinMode(LEFT_PWM_PIN, OUTPUT);
  pinMode(RIGHT_A_PIN, OUTPUT);
  pinMode(RIGHT_B_PIN, OUTPUT);
  pinMode(RIGHT_PWM_PIN, OUTPUT);
  stopAll();

  Serial.begin(9600);
  Serial.println("WSAD motor control ready (9600 baud)");
  Serial.println("W=forward S=reverse A=left D=right X/space=stop");
  Serial.println("Motion continues until the next command.");
}

void loop() {
  while (Serial.available() > 0) {
    char command = Serial.read();
    if (command >= 'a' && command <= 'z') {
      command -= 'a' - 'A';
    }

    switch (command) {
      case 'W':
        drive(1, 1);
        Serial.println("Forward");
        break;
      case 'S':
        drive(-1, -1);
        Serial.println("Reverse");
        break;
      case 'A':
        drive(-1, 1);
        Serial.println("Turn left");
        break;
      case 'D':
        drive(1, -1);
        Serial.println("Turn right");
        break;
      case 'X':
      case ' ':
        stopAll();
        Serial.println("Stopped");
        break;
      case '\r':
      case '\n':
        break;
      default:
        Serial.println("Unknown command: use W/S/A/D or X to stop");
        break;
    }
  }
}
