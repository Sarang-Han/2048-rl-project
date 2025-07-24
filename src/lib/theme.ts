export const theme = {
  colors: {
    primary: {
      background: '#faf8ef',
      text: '#776e65',
      textSecondary: '#8f7a66'
    },
    board: {
      background: '#bbada0',
      cellEmpty: '#cdc1b4',
      border: '#8f7a66',
      frame: '#9c8b7d'
    },
    controls: {
      background: '#ffffff',
      border: '#bbada0',
      button: {
        primary: '#8f7a66',
        primaryHover: '#776e65',
        danger: '#f65e3b',
        dangerHover: '#e04e2b',
        secondary: '#bbada0',
        secondaryHover: '#a89a82'
      }
    },
    status: {
      success: '#22c55e',
      warning: '#f59e0b',
      error: '#ef4444',
      info: '#3b82f6'
    }
  },
  spacing: {
    boardSize: 440,
    gap: 12
  },
  borders: {
    thin: '2px solid',
    medium: '3px solid'
  },
  shadows: {
    tile: {
      main: '0 2px 4px rgba(119, 110, 101, 0.25)',
      hover: '0 3px 6px rgba(119, 110, 101, 0.35)'
    }
  }
} as const;