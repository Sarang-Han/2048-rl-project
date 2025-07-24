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
      frame: '#9c8b7d',
      cellShadow: '#b0a491',
      cellHighlight: '#d8cfc1',
      tileShadow: 'rgba(119, 110, 101, 0.25)'
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
    padding: 24,
    gap: 12,
    borderRadius: {
      small: '8px',
      medium: '12px',
      large: '16px'
    }
  },
  borders: {
    thin: '2px solid',
    medium: '3px solid',
    thick: '4px solid'
  },
  animation: {
    fast: '150ms',
    normal: '300ms',
    slow: '500ms'
  },
  // 🔥 프레임과 셀 구분하여 그림자 효과 정의
  shadows: {
    frame: {
      // 🔥 프레임이 밖으로 튀어나온 느낌
      raised: `
        0 1px 0 rgba(255, 255, 255, 0.3),
        0 -1px 0 rgba(0, 0, 0, 0.2),
        1px 0 0 rgba(255, 255, 255, 0.2),
        -1px 0 0 rgba(0, 0, 0, 0.15),
        0 4px 8px rgba(119, 110, 101, 0.2)
      `
    },
    cell: {
      // 🔥 셀은 안으로 들어간 느낌 (기존 유지)
      inset: 'inset 0 1px 0 rgba(255, 255, 255, 0.35), inset 0 -1px 0 rgba(0, 0, 0, 0.2)',
      top: 'inset 0 1px 0 rgba(0, 0, 0, 0.2)',
      bottom: 'inset 0 -1px 0 rgba(255, 255, 255, 0.35)'
    },
    tile: {
      main: '0 2px 4px rgba(119, 110, 101, 0.25)',
      hover: '0 3px 6px rgba(119, 110, 101, 0.35)'
    }
  }
} as const;