#include <Servo.h>

String str, text, strr;
char buf;
bool action = false, distant = false, reader = false, check = false;
long int TimeCounter = 0;
int counter = 0, randd = 0, wantDist = 0, wantTurn = 1500, wantAngle = 90;

Servo ESC;
Servo Wheel;

const int CENTER = 86;
const int MIN_ANGLE = 44;
const int MAX_ANGLE = 128;

void StartTimer()
{
  cli();
  TCCR2A = 0;
  TCCR2B = 0;

  TIMSK2 |= (1 << TOIE2);                     // прерывание по переполнению
  TCCR2B |= (1 << CS20);                      // переполнение будет происходить каждую 0,0041 секунды
  sei();
}

ISR (TIMER2_OVF_vect)   // прерывание по переполнению
{
  TimeCounter++; 
}

void SpeedTest()
{ 
  counter++;
}

void setup() 
{
  Serial.begin(115200);
  StartTimer();
    
  ESC.attach(7);                           // Подключаем мотор
  ESC.writeMicroseconds(1500);
  attachInterrupt(0, SpeedTest, FALLING);   // Создаем прерывание для энкодера

  Wheel.attach(8);
  Wheel.write(CENTER);
  
}

void loop() {
  while(Serial.available())
  {
   buf = Serial.read();
   
   if (buf == '*')
   {
      reader = true;
   }
   if (reader)
    { 
      str = str + buf;
    }
   
   if ((buf == '|') and (reader == true))
   {
      reader = false;
      action = true;
      break;
   }
  }

  if (action)
  {  
    if (str.startsWith("*GO"))
    {
      text = str.substring(4,8);
      wantTurn = text.toInt();
      ESC.writeMicroseconds(wantTurn);
      Serial.println("GO");
      digitalWrite(LED_BUILTIN, LOW); 
      str = "";
    }

    if (str.startsWith("*STOP"))
    {
      ESC.writeMicroseconds(1500);
      Serial.write("STOP");
      str = "";
    }

    if (str.startsWith("*CHECK"))
    {
      if (check == true)
      {
        Serial.write("end dist");
        check = false;
      }
      else
      {
        Serial.write("no dist");
      }
      str = "";
    }
      
    if (str.startsWith("*DIST"))
    {
      text = str.substring(6,10);
      wantDist = text.toInt();
      //Serial.println(wantDist);
      distant = true;
      TimeCounter = 0;
      str = "";
      Serial.write("start");
      counter = 0;
    }
    
    if (str.startsWith("*ANGLE"))
    {
      text = str.substring(7,10);
      wantAngle = text.toInt();
      // ОГРАНИЧЕНИЯ ПО УГЛАМ!
      if (wantAngle < MIN_ANGLE) {wantAngle = MIN_ANGLE;}
      if (wantAngle > MAX_ANGLE) {wantAngle = MAX_ANGLE;}
      Wheel.write(wantAngle);
      Serial.write("angle");
      str = "";
    }
    
    action = false;
  }
  if (distant)
  {
    if (counter >= wantDist)
    {
      ESC.writeMicroseconds(1500);
      distant = false;
      check = true;
      TimeCounter = 0;  
    }
  }
  
}
