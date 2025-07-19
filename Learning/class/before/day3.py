# ------------------------------------------------------
# 예외 처리
# ------------------------------------------------------
try:
    num = int(input("숫자를 입력하세요: "))
    print(10 / num)
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다.")
except ValueError:
    print("숫자를 입력해야 합니다.")
else:
    print("계산 성공")
finally:
    print("프로그램 종료")
    
# 위의 프로그램에서 0으로 나누려다가 오류가 발생한 경우에 이용자에게
# 다시 숫자를 입력하게 하려면 어떻게 수정해야 할까요?
# 오류가 없으면 계산식을 표시하고 프로그램을 종료한다

while True:
    try:
        num = int(input("숫자를 입력하세요: "))
        answer = 10/num
    except ZeroDivisionError:
        print("0으로 나눌 수 없습니다.")
    except ValueError:
        print("숫자를 입력해야 합니다.")
    else:
        print("{} / {} = {}".format(10, num, answer))
        break

print("프로그램 종료")


# 함수내의 변수는 함수 스코프를 갖는다
# 변수 Scope의 이해
def scope_test():
    idx = 0
    while True:
        if idx != 0:
            print(count, end=" ")
        idx = idx+1
        count = idx
        if count==10:
            break
scope_test()


# ------------------------------------------------------
# 제너레이터 & 이터레이터
# ------------------------------------------------------
def line_reader(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            yield line.strip()
            print('Generator 리턴')

from collections.abc import Iterable

gen = line_reader("Learning/class/before/employee.csv")
gen   # <generator object line_reader at 0x7eab913e09a0>
print(isinstance(gen, Iterable))   # True

gen.__next__()  # generator의 yield 값

next(gen)   # 위의 방법보다 일반적인 호출 방법
next(gen)
next(gen)   # StopIteration. 다시 처음부터 실행하려면 Generator를 다시 생성해야 함


# 루프에서 제너레이터 사용
#for line in line_reader("/content/drive/MyDrive/Python_AI/YOLO/Codes/Python Advanced/employee.csv"):
#    print(line)



# ------------------------------------------------------
# 데코레이터
# ------------------------------------------------------
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print("실행 시간:", time.time() - start)
        return result
    return wrapper

@timer   # 함수의 실행시 경과시간 자동 측정
def slow_function():
    time.sleep(1)
    return "완료!"

slow_function()



# ------------------------------------------------------
# __call__() 메서드 이해와 활용
# ------------------------------------------------------
# 인스턴스를 함수처럼 사용할 수 있는 클래스를 구현한다.
# 🔹 __call__ 메서드 실습 예제

class Multiplier:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, value):   # self 외에 임의의 파라미터 선언 가능
        return self.factor * value

# 인스턴스를 생성하고 함수처럼 호출
triple = Multiplier(3)
result = triple(10)  # → triple.__call__(10)
print("3배 결과:", result)

# 복수 인스턴스를 각각 함수처럼 사용
double = Multiplier(2)
print("2배 결과:", double(7))



# 임의의 클래스를 선언하고 __call__() 함수를 정의하여 아래의 기능을 완성해보세요
# 객체를 호출하여 두수의 덧셈식을 출력한다
class Addition:
    def __init__(self, num1, num2):
        self.num1 = num1
        self.num2 = num2

    def __call__(self):
        result = "{} + {} = {}".format(self.num1, self.num2, self.num1+self.num2)
        print(result)
        return result

add = Addition(3, 5)
add()


# 인자(실인자, 가인자)
# 실인자(Arguments) : 함수에 전달되는 실제 값
# 가인자(Parameter) : 함수에 전달되는 실제 값을 저장할 변수
# 가변인자(Variable Argument) 함수
# 가변인자 함수:미리 정해지지 않은 개수의 인자를 받을 수 있도록 해주는 기능
def process_data(*args, **kwargs):
    """
    가변 위치 인자 (*args)와 가변 키워드 인자 (**kwargs)를 받는 함수 예시
    """
    print('args의 자료형=', type(args))    # args의 자료형= <class 'tuple'>
    print("받은 위치 인자 (args):", args)
    print("받은 키워드 인자 (kwargs):", kwargs)
    print("-" * 20)

process_data(1,2,3)   # Positional Arguments
process_data(num=11, name='Smith', phone='010-1234-5678')   # Keyword Arguments
process_data(1,2,3, num=11, name='Smith', phone='010-1234-5678')

nums = [1,2,3]
process_data(nums)   # 리스트가 그대로 전달되어 tuple의 원소로 사용된다
process_data(*nums)  # Argument Unpacking

n1, n2, n3 = nums    # Sequence Unpacking



import random
rd = random.randint(1,10)  # 
# rd에 저장된 정수의 수만큼 덧셈을 하고 그 결과값을 출력하는 함수를 작성한다
# 가변인자를 가진 덧셈함수를 선언하고 임의의 갯수의 정수를 전달하여 계산하게 한다
def add(*args):
    sum = 0
    for i in range(len(args)):
        sum += args[i]
    print('합산결과=', sum)

nums = []
for i in range(rd):
    nums.append(random.randint(1,10))
print(nums)

add(*nums)



import numpy
arr1 = numpy.array([1,2,3,4,5])
type(arr1)   # numpy.ndarray (numpy배열)
arr2 = numpy.array([5,4,3,2,1])
arr1 + arr2