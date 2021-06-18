def linguistics(h, g, L):
  def combinations(h, g, L):
    r = []
    result = []
    Len = len(h)
    for n in range(pow(Len, L)):
      tmp = ''
      for i in range(L):
        tmp = tmp + h[n % Len] + ' '
        n //= Len
      tmp = tmp[:-1]
      r.append(tmp)

    for i in range(len(r)):
      for j in range(len(g)):
        tmp = ''
        tmp = tmp + r[i] + " " + g[j]
        result.append(tmp)
    return result

  linguisticValues = g
  for i in range(L):
    linguisticValues = linguisticValues + combinations(h, g, i+1)
  return linguisticValues


def HedgeAlgebra(fmSmall, uVery, uMore, length):
    h = ['Very', 'More', 'Possible']
    g = ['Large', 'Small']
    l = length - 1;
    languages = linguistics(h, g, l)
    fmLarge = 1 - fmSmall
    uPossible = 1 - uVery - uMore

    def sig(x):
        if x == "Large" or x == "Very" or x == "More":
            return 1
        else:
            return -1

    def sig2(x, y):
        return sig(x) * sig(y)

    def fmGenerator(x):
        if x == "Large":
            return fmLarge
        else:
            return fmSmall

    def uHedge(x):
        if x == "Very":
            return uVery
        elif x == "More":
            return uMore
        else:
            return uPossible

    def fm(x):
        x = x.split(' ')
        fm = 1
        for i in range(len(x)):
            if i == (len(x) - 1):
                fm = fm * fmGenerator(x[i])
            else:
                fm = fm * uHedge(x[i])
        return fm

    def alpha():
        return uPossible

    def beta(x):
        if x == 'Very':
            return uMore
        else:
            return 0

    def sign(x):
        x = x.split(' ')
        if len(x) > 2:
            y = ''
            for i in range(len(x)):
                if i > 0:
                    y = y + x[i] + ' '
            y = y[:-1]
            signX = sig2(x[0], x[1]) * sign(y)
        elif len(x) == 2:
            signX = sig(x[0]) * sig(x[1])
        else:
            signX = sig(x[0])
        return signX

    def v(x):
        if x == "Large":
            vX = fmGenerator('Small') + alpha() * fmGenerator('Large')
            return vX
        elif x == "Small":
            vX = (1 - alpha()) * fmGenerator('Small')
            return vX
        else:
            signX = sign(x)
            x = x.split(' ')
            y = ''
            for i in range(len(x)):
                if i > 0:
                    y = y + x[i] + ' '
            y = y[:-1]
            signY = sign(y)
            vY = v(y)
            b = beta(x[0])
            fmY = fm(y)
            a = alpha()
            fmX = fm(x[0] + ' ' + y)
            if signX == 1 and signY == 1:
                vX = vY + b * fmY + a * fmX
            elif signX == 1 and signY == -1:
                vX = vY + b * fmY + (1 - a) * fmX
            elif signX == -1 and signY == 1:
                vX = vY - b * fmY - (1 - a) * fmX
            else:
                vX = vY - b * fmY - a * fmX
            return vX

    result = [0, 1]
    for i in range(len(languages)):
        r = round(v(languages[i]), 4)
        result.append(r)
        result.sort()
    return result

