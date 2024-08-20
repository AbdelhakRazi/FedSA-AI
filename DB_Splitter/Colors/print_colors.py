from  .console_colors import colors

def print_color(*strs_, color_=colors.Color_Off, sep=' ', end='\n'):
    st = ""
    #     print(*strs_)
    for s in strs_:
        #         print(s)
        st += str(s) + sep
    st = st.strip()

    print(color_, st, colors.Color_Off)


def print_orange(*strs_, color_=colors.Color_Off, sep=' ', end='\n'):
    print_color(*strs_, color_=colors.Yellow, sep=' ', end='\n')


def print_blue(*strs_, color_=colors.Color_Off, sep=' ', end='\n'):
    print_color(*strs_, color_=colors.Blue, sep=' ', end='\n')


def print_cyne(*strs_, color_=colors.Color_Off, sep=' ', end='\n'):
    print_color(*strs_, color_=colors.Cyan, sep=' ', end='\n')


def print_purple(*strs_, color_=colors.Color_Off, sep=' ', end='\n'):
    print_color(*strs_, color_=colors.Purple, sep=' ', end='\n')


def print_red(*strs_, color_=colors.Color_Off, sep=' ', end='\n'):
    print_color(*strs_, color_=colors.Red, sep=' ', end='\n')
