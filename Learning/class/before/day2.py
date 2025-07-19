# ------------------------------------------------------
# 클래스 기초
# ------------------------------------------------------
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        if amount > self.balance:
            return "잔액 부족"
        self.balance -= amount
        return self.balance

# 인스턴스 생성 및 사용
account = BankAccount("홍길동", 1000)
print(account.deposit(500))
print(account.withdraw(200))


# ------------------------------------------------------
# 상속시에 자식 클래스에서 부모 클래스의 속성을 초기화하는 예
# ------------------------------------------------------
class Parent:
    def __init__(self, name):
        self.name = name
        print(f"Parent 생성자 호출: name = {self.name}")

class Child(Parent):
    def __init__(self, name, age):
        super().__init__(name)  # 부모 생성자 호출
        # super()는 self를 랩핑하여 기능확장하여 부모 클래스의 메소드 호출 가능하도록 한 Proxy 오브젝트
        self.age = age
        print(f"Child 생성자 호출: age = {self.age}")

# 테스트
child = Child("홍길동", 20)



# ------------------------------------------------------
# CSV, JSON 파일 실전
# Comma Separated Value, Javascript Standard Object Notation
# ------------------------------------------------------
import csv
import json

# CSV 저장
data = [["이름", "나이"], ["홍길동", 30], ["김영희", 25]]

file_path = "Learning/class/before/"

with open(file_path+"people.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(data)

# CSV 읽기
with open(file_path+"people.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        name, age = row                   # unpack
        print(f"{name}\t{age}")           # f-string
        #print("{}\t{}".format(name,age)) # format()
        #print("%s\t%s"%(name,age))       # %s

# JSON 저장
person = {"name": "홍길동", "age": 30, "email": "hong@example.com"}
with open(file_path+"person.json", "w", encoding="utf-8") as f:
    json.dump(person, f, ensure_ascii=False, indent=2)

# JSON 읽기
with open(file_path+"person.json", "r", encoding="utf-8") as f:
    person_data = json.load(f)
    print(person_data)
    

#--------------------------------------------------------------------------------
#클래스, 파일 스트림, 컨테이너 종합실습
#기본적인 CRUD(Create, Read, Update, Delete, 검색)
#키보드 입력
#사원정보 관리 시스템
#추가(a), 목록(s), 수정(u), 삭제(d), 검색(f), 종료(x) :

## 클래스, 파일 스트림, 컨테이너 종합실습
# * 기본적인 CRUD(Create, Read, Update, Delete, 검색) 
# * 키보드 입력
# * 사원정보 관리 시스템
# * 추가(a), 목록(s), 수정(u), 삭제(d), 검색(f), 종료(x) : 
# * a : 이용자로부터 사번, 이름, 부서번호, 전화번호 입력 및 csv 파일에 추가
#   - 추가할 때는 파일 모드를 "a" 로 지정
#   - 파일명 : employee.csv
# * s : 사원 목록을 화면에 표시
# * u : 수정할 사번과 새 전화번호를 입력 받아서 기존 데이터 변경
#   - 기존 데이터를 모두 로드하여 리스트에 저장하고 수정대상 정보를 찾아 변경
#   - 메모리에서 변경된 데이터를 다시 employee.csv 파일에 덮어쓰기
# * d : 삭제대상 사번을 입력 받아서 해당 사원 정보 삭제
#   - 기존 데이터를 모두 로드하여 리스트에 저장하고 수정대상 정보를 찾아 삭제
#   - 메모리에서 삭제된 데이터를 다시 employee.csv 파일에 덮어쓰기
# * f : 검색하려는 사원 번호를 입력하여 검색된 사원 정보를 화며에 표시
# * x : 프로그램 메인 루프 종료
# #--------------------------------------------------------------------------------

while True:
    menu = input("추가(a), 목록(s), 수정(u), 삭제(d), 검색(f), 종료(x) :")
    if menu=='a':
        pass
    elif menu=='s':
        pass
    elif menu=='u':
        pass
    elif menu=='d':
        pass
    elif menu=='f':
        pass
    elif menu=='x':
        break
    else:
        print("잘못된 메뉴 선택")

print('프로그램 종료됨')

# 사원정보 추가/목록
while True:
    menu = input("추가(a), 목록(s), 수정(u), 삭제(d), 검색(f), 종료(x) :")
    if menu=='a':
        eno = input('사번:')
        ename = input('이름:')
        dno = input('부서번호:')
        phone = input('전화번호:')
        with open(file_path+"employee.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([eno, ename, dno, phone])
            print('사원정보 추가 성공')
    elif menu=='s':
        with open(file_path+"employee.csv", newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                eno, ename, dno, phone = row
                print(f"{eno}\t{ename}\t{dno}\t{phone}")
    elif menu=='u':
        pass
    elif menu=='d':
        pass
    elif menu=='f':
        pass
    elif menu=='x':
        break
    else:
        print("잘못된 메뉴 선택")

print('프로그램 종료됨')


class Employee:
    def __init__(self, eno, ename, dno, phone):
        self.eno = eno
        self.ename = ename
        self.dno = dno
        self.phone = phone
    def printRow(self):
        row = "{}\t{}\t{}\t{}".format(self.eno, self.ename, 
                                      self.dno, self.phone)
        print(row)
        
        
        
class Employee:
    def __init__(self, info):
        eno, ename, dno, phone = info
        self.eno = eno
        self.ename = ename
        self.dno = dno
        self.phone = phone
        
    def printRow(self):
        row = "{}\t{}\t{}\t{}".format(self.eno, self.ename, 
                                      self.dno, self.phone)
        print(row)

    def saveRow(self):
        with open(file_path+"employee.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([self.eno, self.ename, self.dno, self.phone])
            print('사원정보 추가 성공')


def inputEmp():
    eno = input('사번:')
    ename = input('이름:')
    dno = input('부서번호:')
    phone = input('전화번호:')
    return Employee([eno, ename, dno, phone])
    

while True:
    menu = input("추가(a), 목록(s), 수정(u), 삭제(d), 검색(f), 종료(x) :")
    if menu=='a':
        inputEmp().saveRow()
    elif menu=='s':
        with open(file_path+"employee.csv", newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                Employee(row).printRow()
    elif menu=='u':
        pass
    elif menu=='d':
        pass
    elif menu=='f':
        pass
    elif menu=='x':
        break
    else:
        print("잘못된 메뉴 선택")

print('프로그램 종료됨')


def loadEmps():
    emps = []
    with open(file_path+"employee.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            emps.append(Employee(row))
    return emps
    
    
def showEmps(emps):
    for emp in emps:
        emp.printRow()

def overwrite(emps):
    try:
        with open(file_path + "employee.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for emp in emps:
                writer.writerow([emp.eno, emp.ename, emp.dno, emp.phone])
        return True
    except Exception as e:
        print(f"파일 덮어쓰기 실패: {e}")
        return False        
        
def deleteEmp():
    emps = loadEmps()
    eno = input('삭제할 사번:')
    deleted = False
    for emp in emps:
        if emp.eno == eno:
            emps.remove(emp)
            if overwrite(emps):
                deleted = True
                print('사원정보 삭제 성공')
            break
    if not deleted:
        print('사원정보 삭제 실패')