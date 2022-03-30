class TwoFactorAltman:
    """
    Z = -0,3877 – 1,0736 * Ктл + 0,579 * (ЗК/П)
    Нормальным считается значение коэффициента 2 и более. Значение ниже 1 говорит о высоком финансовом риске,
    связанном с тем, что предприятие не в состоянии стабильно оплачивать текущие счета.
    Значение более 3 может свидетельствовать о нерациональной структуре капитала.
    """
    def __init__(self, data):
        self.data = data
        self.calculate()

    def calculate(self):
        balance = self.data['balance']

        ktl = balance.get('current1200', 0) / balance.get('current1500', 1)
        borrowed_capital = balance.get('current1400', 0) + balance.get('current1500', 0)
        passive = balance.get('current1700', 1)

        coefficient = -0.3877 - 1.0736 * ktl + 0.579 * (borrowed_capital / passive)

        print('----------------')
        print('Двухфакторная модель Альтмана')
        if coefficient <= 1:
            print('Высокая вероятность банкротства')
        elif coefficient > 3:
            print('Нерациональная структура капитала')
        else:
            print('Нормальное значение коэффициента')
