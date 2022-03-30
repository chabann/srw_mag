class Taffler:
    """
    Модель применима для компаний в форме открытых акционерных обществ, акции которых прошли процедуру публичного
    размещения и торгуются на различных фондовых площадках.
    Z = 0,53X1 + 0,13Х2 + 0,18Х3 + 0,16X4
    """
    def __init__(self, data):
        self.data = data
        self.calculate()

    def calculate(self):
        balance = self.data['balance']

        param1 = balance.get('current2200', 0) / balance.get(1500, 1)
        param2 = balance.get('current1200', 0) / (balance.get(1400, 1) + balance.get(1500, 1))
        param3 = balance.get('current1500', 0) / balance.get(1600, 1)
        param4 = balance.get('current2110', 0) / balance.get(1600, 1)

        coefficient = 0.53 * param1 + 0.13 * param2 + 0.18 * param3 + 0.16 * param4

        print('----------------')
        print('Модель Таффлера (четырехфакторная модель банкротства)')
        if coefficient > 0.3:
            print('Приемлемое финансовое состояние')
        elif coefficient < 0.2:
            print('Высокая вероятность банкротства')
        else:
            print('Нестабильная ситуация, вероятность наступления банкротства организации невелика, но и не исключена')
