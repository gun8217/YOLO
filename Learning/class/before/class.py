import random

class Character:
    def __init__(self, name, hp, attack_power):
        self.name = name
        self.max_hp = hp
        self.hp = hp
        self.attack_power = attack_power
        self.level = 1
        self.exp = 0
        self.potions = 3  # 초기 물약 수량

    def attack(self, target):
        print(f"{self.name}이(가) {target.name}을 공격합니다!")
        target.hp -= self.attack_power
        if target.hp < 0:
            target.hp = 0
        print(f"{target.name}의 남은 HP: {target.hp}")

    def is_alive(self):
        return self.hp > 0

    def gain_exp(self, amount):
        print(f"{self.name}이(가) 경험치 {amount} 획득!")
        self.exp += amount
        self.check_level_up()

    def check_level_up(self):
        required_exp = self.level * 50
        while self.exp >= required_exp:
            self.exp -= required_exp
            self.level += 1
            self.max_hp += 30
            self.attack_power += 10
            self.hp = self.max_hp
            print(f"\n🆙 레벨 {self.level} 도달! HP와 공격력이 상승했습니다!")
            print(f"▶ 현재 HP: {self.hp}, 공격력: {self.attack_power}")
            required_exp = self.level * 50

    def use_potion(self):
        if self.potions > 0:
            heal = random.randint(30, 50)
            self.hp += heal
            if self.hp > self.max_hp:
                self.hp = self.max_hp
            self.potions -= 1
            print(f"\n🧪 {self.name}이(가) 물약을 사용해 {heal} 회복! ▶ 현재 HP: {self.hp}, 남은 물약: {self.potions}")
        else:
            print("\n⚠️ 물약이 없습니다!")

# 몬스터 선택 함수
def get_monster_by_level(level):
    if level < 3:
        return Character("슬라임", 40, 10)
    elif level < 5:
        return Character("고블린", 80, 20)
    else:
        return Character("드래곤", 150, 40)

# 기사 생성
hero = Character("기사", 100, 30)

# 게임 루프
while hero.is_alive():
    monster = get_monster_by_level(hero.level)
    print(f"\n💥 몬스터 등장: {monster.name} (HP: {monster.hp})")

    while monster.is_alive() and hero.is_alive():
        print(f"\n[{hero.name}의 턴] ▶ HP: {hero.hp} / 물약: {hero.potions}")
        print("행동을 선택하세요: [1] 공격  [2] 물약 사용  [3] 퇴각")
        choice = input("👉 입력: ")

        if choice == "1":
            hero.attack(monster)
        elif choice == "2":
            hero.use_potion()
        elif choice == "3":
            print(f"\n🚪 {hero.name}이(가) 퇴각했습니다. 게임 종료...")
            exit()
        else:
            print("\n❌ 잘못된 입력입니다. 다시 선택하세요.")
            continue

        if monster.is_alive():
            monster.attack(hero)
            if not hero.is_alive():
                print(f"\n💀 {hero.name}이(가) 사망했습니다. 게임 오버...")
                break
        else:
            print(f"\n🏆 {monster.name} 처치 성공!")
            hero.gain_exp(random.randint(40, 60))

if hero.is_alive():
    print(f"\n🎉 {hero.name}은 모든 전투에서 승리했습니다!")
