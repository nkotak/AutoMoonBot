from typing import List
import numpy as np

class Portfolio:
    def __init__(
        self,
        fiat: str,
        tradables: List[str],
    ) -> None:
        self.index_map = {t: i for t, i in enumerate(tradables)}
        print(f"DEBUG: fiat='{fiat}'")
        print(f"DEBUG: tradables={tradables}")
        print(f"DEBUG: index_map={self.index_map}")
        self.fiat = self.index_map[fiat]

try:
    p = Portfolio(fiat='USD', tradables=['USD', 'AAPL', 'GOOGL', 'MSFT'])
    print("Success")
except Exception as e:
    print(f"Error: {e}")
