class NonManufacturingAltman:
    """
    Z = 6,56Х1 + 3,26Х2 + 6,72Х3 + 1,05Х4
    Х1 – Оборотный капитал / Активы = («1200» - «1500») / «1600»;
    Х2 – Нераспределенная прибыль / Активы = «1370» / «1600»;
    Х3 – Прибыль до налогообложения/ Активы = «2300» / «1600»;
    Х4 – Собственный капитал / Обязательства = «1300» / («1400» + «1500»)
    """
    def __init__(self, data):
        self.data = data
        self.calculate()

    def calculate(self):
        balance = self.data['balance']

        param1 = (balance.get('current1200', 0) - balance.get('current1500', 0)) / balance.get(1600, 1)
        param2 = balance.get('current1370', 0) / balance.get(1600, 1)
        param3 = balance.get('current2300', 0) / balance.get(1600, 1)
        param4 = balance.get('current1300', 0) / (balance.get(1400, 1) + balance.get(1500, 1))

        coefficient = 6.56 * param1 + 3.26 * param2 + 6.72 * param3 + 1.05 * param4

        print('----------------')
        print('Модель Альтмана для непроизводственных компаний')
        if coefficient <= 1.1:
            print('Высокая вероятность банкротства')
        elif coefficient >= 2.6:
            print('Нестабильная ситуация, вероятность наступления банкротства организации невелика, но и не исключена')
        else:
            print('Низкая вероятность банкротства ')
