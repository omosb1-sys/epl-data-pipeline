
"""
[Types as Proofs: Logical integrity Demo]
Inspired by Evan Moon's 'Types as Proofs'
=========================================
이 코드는 파이썬의 Type Hinting을 단순한 힌트가 아닌 '논리적 명제'로 대하는
Antigravity Rule 38의 철학을 보여주는 시뮬레이션입니다.
"""

from typing import TypeVar, Callable, Generic, Any
from dataclasses import dataclass

# 논리적 엔티티(명제) 정의
@dataclass(frozen=True)
class Flour:
    weight: int

@dataclass(frozen=True)
class Bread:
    kcal: int

# [Rule 38.1] 함수 타입은 "A이면 B이다"라는 명제입니다.
# make_bread: Flour -> Bread (밀가루가 있으면 빵을 만들 수 있다)
def make_bread(flour: Flour) -> Bread:
    """
    이 구현체(Proof)는 Flour input이 들어왔을 때 Bread를 반환할 수 있음을 증명합니다.
    """
    # 만약 여기서 Bread가 아닌 다른 것을 반환한다면, 그것은 논리적 계약 위반(Type Error)입니다.
    return Bread(kcal=flour.weight * 4)

def test_logical_integrity():
    # 명제: Flour가 존재한다.
    my_flour = Flour(weight=100)
    
    # 추론: 명제(make_bread)가 참이고 전건(my_flour)이 참이면, 후건(my_bread)도 참이다.
    my_bread: Bread = make_bread(my_flour)
    
    print(f"✅ [Logical Proof] 명제 완료: {my_flour} -> {my_bread}")

    # [Rule 38.2] 'any'를 통한 증명 파괴 시뮬레이션 (경고)
    broken_proof: Any = "Not Flour"
    # 아래 코드는 런타임에 문제를 일으킬 수 있는 '논리적 모순'의 복선입니다.
    # my_bread_broken: Bread = make_bread(broken_proof) # static type checker에서 잡힘

if __name__ == "__main__":
    test_logical_integrity()
